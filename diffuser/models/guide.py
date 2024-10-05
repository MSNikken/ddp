# The following code is adapted from Motion Planning Diffusion
# Learning and Planning of Robot Motions with Diffusion Models,
# by João Carvalho, An T. Le, Mark Baier, Dorothea Koert and Jan Peters
# from https://github.com/jacarvalho/mpd-public and dependency https://github.com/anindex/motion_planning_baselines

import torch
from torch import nn
import pypose as pp


def guide_gradient_steps(
    x,
    guide=None,
    n_guide_steps=1, scale_grad_by_std=False,
    model_var=None,
):
    for _ in range(n_guide_steps):
        grad_scaled = guide(x)

        if scale_grad_by_std:
            grad_scaled = model_var * grad_scaled

        x = x + grad_scaled
    return x


class GuideManagerTrajectories(nn.Module):

    def __init__(self, cost, normalizer, clip_grad=False, clip_grad_rule='norm', max_grad_norm=1., max_grad_value=0.1,
                 interpolate_trajectories_for_collision=False,
                 num_interpolated_points_for_collision=128):
        super().__init__()
        self.cost = cost
        self.normalizer = normalizer

        self.interpolate_trajectories_for_collision = interpolate_trajectories_for_collision
        self.num_interpolated_points_for_collision = num_interpolated_points_for_collision

        self.clip_grad = clip_grad
        self.clip_grad_rule = clip_grad_rule
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value

    def forward(self, x_pos_normalized):
        x_pos = x_pos_normalized.clone()
        with torch.enable_grad():
            x_pos.requires_grad_(True)

            # unnormalize x
            # x is normalized, but the guides are defined on unnormalized trajectory space
            x_pos = self.normalizer.unnormalize(x_pos, 'observations')

            if self.interpolate_trajectories_for_collision:
                # finer interpolation of trajectory for better collision avoidance
                x_interpolated = interpolate_points_v1(x_pos, num_interpolated_points=self.num_interpolated_points_for_collision)
            else:
                x_interpolated = x_pos

            cost_l, weight_grad_cost_l = self.cost(x_pos, x_interpolated=x_interpolated, return_invidual_costs_and_weights=True)
            grad = 0
            for cost, weight_grad_cost in zip(cost_l, weight_grad_cost_l):
                if torch.is_tensor(cost):
                    # y.sum() is a surrogate to compute gradients of independent quantities over the batch dimension
                    # x are the support points. Compute gradients wrt x, not x_interpolated
                    grad_cost = torch.autograd.grad([cost.sum()], [x_pos], retain_graph=True)[0]

                    # clip gradients
                    grad_cost_clipped = self.clip_gradient(grad_cost)

                    # zeroing gradients at start and goal
                    grad_cost_clipped[..., 0, :] = 0.
                    grad_cost_clipped[..., -1, :] = 0.

                    # combine gradients
                    grad_cost_clipped_weighted = weight_grad_cost * grad_cost_clipped
                    grad += grad_cost_clipped_weighted

        # gradient ascent
        grad = -1. * grad
        return grad

    def clip_gradient(self, grad):
        if self.clip_grad:
            if self.clip_grad_rule == 'norm':
                return self.clip_grad_by_norm(grad)
            elif self.clip_grad_rule == 'value':
                return self.clip_grad_by_value(grad)
            else:
                raise NotImplementedError
        else:
            return grad

    def clip_grad_by_norm(self, grad):
        # clip gradient by norm
        if self.clip_grad:
            grad_norm = torch.linalg.norm(grad + 1e-6, dim=-1, keepdims=True)
            scale_ratio = torch.clip(grad_norm, 0., self.max_grad_norm) / grad_norm
            grad = scale_ratio * grad
        return grad

    def clip_grad_by_value(self, grad):
        # clip gradient by value
        if self.clip_grad:
            grad = torch.clip(grad, -self.max_grad_value, self.max_grad_value)
        return grad


class CostComposite(object):
    def __init__(self, cost_list, weights=None):
        self.cost_l = cost_list
        self.cost_weights = weights if weights is not None else [1.0] * len(cost_list)

    def __call__(self, *args, **kwargs):
        cost_l = []
        for cost in self.cost_l:
            cost_eval = cost(*args, **kwargs)
            cost_l.append(cost_eval)
        return cost_l, self.cost_weights
