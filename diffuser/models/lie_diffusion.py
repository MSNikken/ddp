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
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
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

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
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
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

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

    def q_posterior(self, H0: pp.se3_type, Hk: pp.se3_type, k):
        tangent_shape = (*H0.shape[0:2], 6)
        posterior_mean = pp.se3(
                extract(self.posterior_mean_coef1, k, tangent_shape) * H0 +
                extract(self.posterior_mean_coef2, k, tangent_shape) * Hk
        )
        posterior_variance = extract(self.posterior_variance, k, tangent_shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, k, tangent_shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, Hk, cond, k, returns=None):
        if self.returns_condition:
            H_k0_cond = self.model(Hk, cond, k, returns, use_dropout=False)
            H_k0_uncond = self.model(Hk, cond, k, returns, force_dropout=True)
            H_k0 = H_k0_uncond + self.condition_guidance_w * (H_k0_cond - H_k0_uncond)
        else:
            H_k0 = self.model(Hk, cond, k)

        H0_recon = pp.Log(pp.Exp(pp.se3(H_k0)) @ pp.Exp(pp.se3(Hk)).Inv())
        #k = k.detach().to(torch.int64)

        if self.clip_denoised:
            H0_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            H0_recon, Hk, k)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, Hk, cond, k, returns=None):
        epsilon = torch.randn(Hk.shape, device=Hk.device)
        model_mean, posterior_variance, _ = self.p_mean_variance(Hk, cond, k, returns)

        return model_mean + pp.se3(torch.sqrt(posterior_variance * epsilon))

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        noise = 0.5 * torch.randn(shape, device=device)
        H_k = pp.se3(noise)
        H_k = apply_conditioning(H_k, cond, 0)

        if return_diffusion: diffusion = [H_k]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            H_k = self.p_sample(H_k, cond, timesteps, returns)
            H_k = apply_conditioning(H_k, cond, 0)

            progress.update({'k': i})

            if return_diffusion: diffusion.append(H_k)

        progress.close()

        if return_diffusion:
            return H_k, torch.stack(diffusion, dim=1)
        else:
            return H_k

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

    def q_sample(self, H0, t, noise=None, gamma=1):
        if noise is None:
            noise = torch.randn(H0.shape, device=H0.device)

        # sample = (
        #         extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        #         extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        # )

        perturb = pp.Exp(pp.se3(gamma * extract(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise))
        interp = interpolate_sqrt_alphas_cumprod(pp.Exp(H0), extract(self.sqrt_alphas_cumprod, t, noise.shape))
        return perturb @ interp

    def p_losses(self, H, V, cond, t, returns=None):
        H = pp.se3(H)
        H_noisy = pp.Log(self.q_sample(H0=H, t=t))
        H_noisy = apply_conditioning(H_noisy, cond, 0)

        H_rel_recon = self.model(H_noisy.tensor(), cond, t, returns)

        H_rel_hat_recon = pp.se3(H_rel_recon)
        H_rel_hat = pp.Log(pp.Exp(H) @ pp.Exp(H_noisy.Inv()))
        loss, info = self.loss_fn(H_rel_hat_recon, H_rel_hat)

        return loss, info

    def loss(self, x, cond, pose, vel, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        #diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
        diffuse_loss, info = self.p_losses(pose, vel, cond, t, returns)
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
