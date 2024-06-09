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


def kinematic_consistency(x, dt, norm=False):
    x_t = x[..., :-1, :]
    x_t_dt = x[..., 1:, :]

    H_t = pp.se3(x_t[..., :6]).Exp()
    T_t = pp.se3(x_t[..., 6:])
    forward = pp.Exp(T_t * (dt/2)) @ H_t

    H_t_dt = pp.se3(x_t_dt[..., :6]).Exp()
    T_t_dt = pp.se3(x_t_dt[..., 6:])
    backward = pp.Exp(T_t_dt * -(dt/2)) @ H_t_dt

    diff = torch.linalg.vector_norm((forward @ backward.Inv()).Log(), dim=-1)
    consistency = diff
    if norm:
        transition = torch.linalg.vector_norm((H_t @ H_t_dt.Inv()).Log(), dim=-1)
        consistency = diff/transition
    return consistency


def kinematic_pose_consistency(x, norm=False):
    H_t = pp.se3(x[..., :-2, :6]).Exp()
    H_t_1 = pp.se3(x[..., 1:-1, :6]).Exp()
    H_t_2 = pp.se3(x[..., 2:, :6]).Exp()

    forward_midpoint = H_t @ ((H_t.Inv() @ H_t_2).Log() * 0.5).Exp()
    backward_midpoint = H_t_2 @ ((H_t_2.Inv() @ H_t).Log() * 0.5).Exp()

    diff_forward = torch.linalg.vector_norm((H_t_1.Inv() @ forward_midpoint).Log(), dim=-1)
    diff_backward = torch.linalg.vector_norm((H_t_1.Inv() @ backward_midpoint).Log(), dim=-1)
    consistency = diff_forward
    if norm:
        transition_length_sq = torch.linalg.vector_norm(((H_t.Inv() @ H_t_2).Log()*0.5), dim=-1)
        consistency = consistency/transition_length_sq
    return consistency


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
        max_index = weighted_loss.argmax()
        # Search for minimum loss among nonzero weights
        min_index = torch.where((self.k_weights[k] != 0), weighted_loss, 1e10).argmin()
        mean_loss = weighted_loss.mean()
        return mean_loss, {'kin_loss': mean_loss,
                           'max_loss': weighted_loss[max_index],
                           'min_loss': weighted_loss[min_index],
                           'k_max_loss': k[max_index],
                           'k_min_loss': k[min_index],
                           'k_min': k.min(),
                           'k_max': k.max(),
                           }


class KinematicL2(WeightedKinematicLoss):
    def __init__(self, t_weights, k_weights, dt, norm=False):
        super().__init__(t_weights, k_weights)
        self.dt = dt
        self.norm = norm

    def _loss(self, traj):
        score = kinematic_consistency(traj, self.dt, norm=self.norm)
        return F.mse_loss(score, torch.zeros_like(score), reduction='none')


class KinematicPoseL2(WeightedKinematicLoss):
    def __init__(self, t_weights, k_weights, dt, norm=False):
        super().__init__(t_weights, k_weights)
        self.norm = norm

    def _loss(self, traj):
        score = kinematic_pose_consistency(traj, norm=self.norm)
        return F.mse_loss(score, torch.zeros_like(score), reduction='none')


class KinematicLInf(WeightedKinematicLoss):
    def __init__(self, t_weights, k_weights, dt, norm=False):
        super().__init__(t_weights, k_weights)
        self.dt = dt
        self.norm = norm

    def _loss(self, traj):
        score = kinematic_consistency(traj, self.dt, norm=self.norm)
        max, max_i = torch.max(score)
        return F.mse_loss(torch.where(score == max, score, 0), torch.zeros_like(score), reduction='none')


class KinematicPoseLInf(WeightedKinematicLoss):
    def __init__(self, t_weights, k_weights, dt, norm=False):
        super().__init__(t_weights, k_weights)
        self.norm = norm

    def _loss(self, traj):
        score = kinematic_pose_consistency(traj, norm=self.norm)
        max, max_i = torch.max(score)
        return F.mse_loss(torch.where(score == max, score, 0), torch.zeros_like(score), reduction='none')



Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
    'state_l1': WeightedStateL1,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'kinematic_l2': KinematicL2,
    'kinematic_pose_l2': KinematicPoseL2,
    'kinematic_linf': KinematicLInf,
    'kinematic_pose_linf': KinematicPoseLInf
}


def traj_euc2se3(x: torch.Tensor, twist=True):
    traj_pos = x[..., :, :3]
    traj_rot = x[..., :, 3:7]
    traj_rot = torch.nn.functional.normalize(traj_rot, dim=-1)
    traj_pose = pp.SE3(torch.cat([traj_pos, traj_rot], dim=-1)).Log()
    if twist:
        traj_twist = x[..., :, 7:13]
        return torch.cat([traj_pose.tensor(), traj_twist], dim=-1)
    return traj_pose.tensor()
