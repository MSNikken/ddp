import typing
from collections import namedtuple

import numpy as np
import torch
import pypose as pp

Zone = namedtuple('Zone', ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'])


def discounted_trajectory_rewards(traj, zones, discount=1, kin_rel_weight=0, kin_norm=True, kin_l1=False, **kwargs):
    factors = discount ** torch.tensor(range(traj.shape[1]), device=traj.device)
    rew_obst = reward_by_zone(traj, zones, **kwargs)
    rew_kin = torch.zeros_like(rew_obst)
    rew_kin[:, :-2] = -kinematic_pose_consistency(traj, norm=kin_norm)
    if kin_l1:
        mins, idx = torch.min(rew_kin, dim=1)
        rew_kin[...] = 0
        for i, (min_val, min_id) in enumerate(zip(mins, idx)):
            rew_kin[i, min_id] = min_val
    return (factors * (1 - kin_rel_weight) * rew_obst + kin_rel_weight * rew_kin).sum(dim=1)


def reward_by_zone(H, zones: typing.List[Zone], dist_scale=None):
    rewards = torch.zeros(H.shape[:-1], device=H.device, dtype=H.dtype)
    for zone in zones:
        if dist_scale is None:
            mask = ((H[..., 0] > zone.xmin) & (H[..., 0] < zone.xmax) &
                    (H[..., 1] > zone.ymin) & (H[..., 1] < zone.ymax) &
                    (H[..., 2] > zone.zmin) & (H[..., 2] < zone.zmax))
            rewards[mask] = torch.minimum(rewards[mask], torch.tensor([-1.0], device=H.device))
            continue

        dx = torch.clip(torch.maximum(zone.xmin - H[..., 0], H[..., 0] - zone.xmax), min=0)
        dy = torch.clip(torch.maximum(zone.ymin - H[..., 1], H[..., 1] - zone.ymax), min=0)
        dz = torch.clip(torch.maximum(zone.zmin - H[..., 2], H[..., 2] - zone.zmax), min=0)
        dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
        rewards = torch.minimum(rewards, -torch.exp(-4.6 * dist / dist_scale))
    return rewards


def reward_distance_to_end(H, xmin, xmax):
    return -(torch.linalg.vector_norm((H[..., :3] - H[..., -1, :3][..., None, :]), dim=-1) / np.linalg.norm(
        xmax - xmin)) ** 2


def cost_ee(goal, x, **kwargs):
    pos = x[:, :, :3]
    rot = pp.SO3(x[:, :, 3:7])

    goal_pos = goal[..., :3]
    goal_rot = pp.SO3(goal[..., 3:7])

    pos_cost = torch.linalg.vector_norm(pos-goal_pos, dim=-1)**2
    rot_cost = torch.linalg.vector_norm((rot.Inv() @ goal_rot).Log(), dim=-1)
    return torch.sum(pos_cost + rot_cost, dim=-1)


def cost_collision(obsts, x, **kwargs):
    costs = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
    for obst in obsts:
        dx = torch.clip(torch.minimum(x[..., 0] - obst.xmin, obst.xmax - x[..., 0]), min=0)
        dy = torch.clip(torch.minimum(x[..., 1] - obst.ymin, obst.ymax - x[..., 1]), min=0)
        dz = torch.clip(torch.minimum(x[..., 2] - obst.zmin, obst.zmax - x[..., 2]), min=0)
        dist = torch.min(torch.stack([dx, dy, dz]), dim=0)[0]
        costs = torch.maximum(costs, dist)
    return torch.sum(costs, dim=-1)


def kinematic_pose_consistency(H, norm=False, eps=1e-5):
    H_t = pp.SE3(H[..., :-2, :7])
    H_t_1 = pp.SE3(H[..., 1:-1, :7])
    H_t_2 = pp.SE3(H[..., 2:, :7])

    forward_midpoint = H_t @ ((H_t.Inv() @ H_t_2).Log() * 0.5).Exp()
    diff_forward = torch.linalg.vector_norm((H_t_1.Inv() @ forward_midpoint).Log(), dim=-1)
    consistency = diff_forward
    if norm:
        transition_length_sq = torch.linalg.vector_norm(((H_t.Inv() @ H_t_2).Log() * 0.5), dim=-1)
        consistency = consistency / (transition_length_sq + eps)
    return consistency


def rot_from_ref(H, r_ref):
    r_ref = pp.SO3(torch.tensor(r_ref, device=H.device, dtype=H.dtype))
    r = pp.SO3(H[..., 3:7])
    diff = torch.linalg.vector_norm((r_ref @ r.Inv()).Log(), dim=-1)
    return diff


