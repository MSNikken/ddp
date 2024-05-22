import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb
import pypose as pp

import diffuser.utils as utils


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


# -----------------------------------------------------------------------------#
# ------------------------ physical evaluation --------------------------------#
# -----------------------------------------------------------------------------#
def dist_SE3(H1, H2):
    # At identity
    d1 = torch.linalg.vector_norm(pp.Log(H1 * H2.Inv()), dim=-1)
    d2 = torch.linalg.vector_norm(pp.Log(H2 * H1.Inv()), dim=-1)
    # At H1
    d3 = torch.linalg.vector_norm(pp.Log(H1.Inv() * H2), dim=-1)
    # At H2
    d4 = torch.linalg.vector_norm(pp.Log(H2.Inv() * H1), dim=-1)
    # Undefined
    d5 = torch.linalg.vector_norm(pp.Log(H1) - pp.Log(H2), dim=-1)
    return d1


def kinematic_consistency(x, dt):
    x_t = x[..., :-1, :]
    x_t_dt = x[..., 1:, :]

    H_t = pp.se3(x_t[..., :6]).Exp()
    T_t = pp.se3(x_t[..., 6:])
    forward = pp.Exp(T_t * (dt/2)) @ H_t

    H_t_dt = pp.se3(x_t_dt[..., :6]).Exp()
    T_t_dt = pp.se3(x_t_dt[..., 6:])
    backward = pp.Exp(T_t_dt * -(dt/2)) @ H_t_dt

    diff = (forward @ backward.Inv()).Log()
    return torch.sum(diff**2, dim=-1)

    # # Basline dist
    # dist_base = torch.mean(dist1(H_t, H_t_dt))
    # # Forward/backward projection
    # H_forward = pp.Exp(pp.se3(T_t * dt / 2)) * H_t
    # H_backward = pp.Exp(pp.se3(T_t_dt * dt / -2)) * H_t_dt
    # dist_a1 = torch.mean(dist1(H_forward, H_t_dt))
    #
    # H_forward = pp.Exp(pp.se3(H_t.Adj(T_t) * dt / 2)) * H_t
    # H_backward = pp.Exp(pp.se3(H_t_dt.Adj(T_t_dt) * dt / -2)) * H_t_dt
    # dist_a2 = torch.mean(dist1(H_forward, H_t_dt))
    #
    # H_forward = pp.Exp(pp.se3(H_t.Inv().Adj(T_t) * dt / 2)) * H_t
    # H_backward = pp.Exp(pp.se3(H_t_dt.Inv().Adj(T_t_dt) * dt / -2)) * H_t_dt
    # dist_a3 = torch.mean(dist1(H_forward, H_t_dt))
    #
    # # Compare some variants:
    # from diffuser.utils.visualization import plot_trajectory
    # plot_trajectory(H_t)


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}


class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


class WeightedStateL1(WeightedStateLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


class WeightedKinematicLoss(nn.Module):

    def __init__(self, t_weights, k_weights):
        super().__init__()
        self.register_buffer('t_weights', t_weights)
        self.register_buffer('k_weights', k_weights)

    def forward(self, traj, k):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(traj)
        weighted_loss = ((loss * self.t_weights).mean(dim=1)*self.k_weights[k])
        max_loss, max_index = weighted_loss.max()
        return weighted_loss.mean(), {'kin_loss': weighted_loss, 'k_max_loss': max_loss, 'max_loss': k[max_index]}


class KinematicL2(WeightedKinematicLoss):
    def __init__(self, t_weights, k_weights, dt):
        super().__init__(t_weights, k_weights)
        self.dt = dt

    def _loss(self, traj):
        return kinematic_consistency(traj, self.dt)


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
    'state_l1': WeightedStateL1,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'kinematic_l2': KinematicL2
}
