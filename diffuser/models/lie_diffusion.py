import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Union

from diffuser import utils
from diffuser.models.diffusion import ARInvModel
from diffuser.models.helpers import cosine_beta_schedule, Losses, extract, apply_conditioning

import pypose as pp


def interpolate(H1: pp.SE3_type, H2: pp.SE3_type, scale: Union[float, torch.Tensor]):
    return pp.Exp(pp.se3(scale * pp.Log(H2 @ H1.Inv()))) @ H1


def interpolate_sqrt_alphas_cumprod(H, sqrt_alphas_cumprod):
    return interpolate(H, pp.identity_like(H, device=H.device), 1 - sqrt_alphas_cumprod)


class SE3Diffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_diffsteps=1000,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
                 condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_diffsteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_diffsteps = int(n_diffsteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        # TODO add velocity in loss
        dim_weights = torch.ones(self.observation_dim - 6, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x0: pp.se3_type, xk: pp.se3_type, k):
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x0.shape) * x0 +
            extract(self.posterior_mean_coef2, k, x0.shape) * xk
        )
        posterior_variance = extract(self.posterior_variance, k, x0.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, k, x0.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, xk, cond, k, returns=None):
        if self.returns_condition:
            x_k0_cond = self.model(xk, cond, k, returns, use_dropout=False)
            x_k0_uncond = self.model(xk, cond, k, returns, force_dropout=True)
            x_k0 = x_k0_uncond + self.condition_guidance_w * (x_k0_cond - x_k0_uncond)
        else:
            x_k0 = self.model(xk, cond, k)

        H_k0 = pp.Exp(pp.se3(x_k0[..., :6]))
        H_k = pp.Exp(pp.se3(xk[..., :6]))
        H0_recon = pp.Log(H_k0 @ H_k)

        T_k = xk[..., 6:12]
        T_epsilon = x_k0[..., 6:12]
        T0_recon = self.predict_start_from_noise(T_k, k, T_epsilon)
        #k = k.detach().to(torch.int64)

        if self.clip_denoised:
            H0_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            torch.cat([H0_recon, T0_recon], dim=-1), xk, k)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, xk, cond, k, returns=None):
        b, *_, device = *xk.shape, xk.device
        model_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(xk, cond, k, returns)
        noise = 0.5 * torch.randn_like(xk)
        # no noise when t == 0
        nonzero_mask = (1 - (k == 0).float()).reshape(b, *((1,) * (len(xk.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        xk = 0.5 * torch.randn(shape, device=device)
        xk = apply_conditioning(xk, cond, 0)

        if return_diffusion: diffusion = [xk]

        progress = utils.Progress(self.n_diffsteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_diffsteps)):
            diffsteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            xk = self.p_sample(xk, cond, diffsteps, returns)
            xk = apply_conditioning(xk, cond, 0)

            progress.update({'k': i})

            if return_diffusion: diffusion.append(xk)

        progress.close()

        if return_diffusion:
            return xk, torch.stack(diffusion, dim=1)
        else:
            return xk

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, k, noise=None, gamma=1):
        if noise is None:
            H_noise = torch.randn((*x_start.shape[:-1], 6), device=x_start.device)
            T_noise = torch.randn((*x_start.shape[:-1], 6), device=x_start.device)
        else:
            H_noise = noise[..., :6]
            T_noise = noise[..., 6:12]

        H_start = x_start[..., :6]
        T_start = x_start[..., 6:12]

        T_k = (
                extract(self.sqrt_alphas_cumprod, k, T_start.shape) * T_start +
                extract(self.sqrt_one_minus_alphas_cumprod, k, T_start.shape) * T_noise
        )

        perturb = pp.Exp(pp.se3(gamma * extract(self.sqrt_one_minus_alphas_cumprod, k, H_noise.shape) * H_noise))
        interp = interpolate_sqrt_alphas_cumprod(pp.Exp(pp.se3(H_start)), extract(self.sqrt_alphas_cumprod, k, H_noise.shape))
        H_k = pp.Log(perturb @ interp)
        return torch.cat([H_k.tensor(), T_k], dim=-1)

    def p_losses(self, x_start, cond, k, returns=None):
        x_noisy = self.q_sample(x_start=x_start, k=k)
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, k, returns)
        H = pp.Exp(pp.se3(x_start[..., :6]))
        H_noisy = pp.Exp(pp.se3(x_noisy[..., :6]))
        H_rel_recon = pp.se3(x_recon[..., :6])
        H_rel = pp.Log(H @ H_noisy.Inv())
        loss, info = self.loss_fn(H_rel_recon, H_rel)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        k = torch.randint(0, self.n_diffsteps, (batch_size,), device=x.device).long()
        #diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
        diffuse_loss, info = self.p_losses(x, cond, k, returns)
        # Calculating inv loss
        # x_t = x[:, :-1, self.action_dim:]
        # a_t = x[:, :-1, :self.action_dim]
        # x_t_1 = x[:, 1:, self.action_dim:]
        # x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        # x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        # a_t = a_t.reshape(-1, self.action_dim)
        # if self.ar_inv:
        #     inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
        # else:
        #     pred_a_t = self.inv_model(x_comb_t)
        #     inv_loss = F.mse_loss(pred_a_t, a_t)

        # loss = (1 / 2) * (diffuse_loss + inv_loss)
        loss = diffuse_loss

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
