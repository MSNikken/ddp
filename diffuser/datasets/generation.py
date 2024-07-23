import typing
from collections import namedtuple

import numpy as np
import torch, pypose as pp
from matplotlib import pyplot as plt

try:
    from ..utils.visualization import plot_trajectory
    from ..models.helpers import dist_SE3, kinematic_consistency, kinematic_pose_consistency
except ImportError:
    from diffuser.utils.visualization import plot_trajectory, plot_trajectory_2d, draw_rectangles
    from diffuser.models.helpers import dist_SE3, kinematic_consistency, kinematic_pose_consistency

Zone = namedtuple('Zone', ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'])


def add_pose_noise(x, std, gamma=100):
    x[..., :6] = (pp.se3(torch.randn((*x.shape[:2], 6), dtype=x.dtype)*(std*gamma)).Exp() @ pp.se3(x[..., :6]).Exp()).Log().tensor()
    return x


def add_t_noise(x, std):
    x[..., 6:] = torch.randn((*x.shape[:2], 6), dtype=x.dtype)*std + x[..., 6:]
    return x


def add_noise(x, sigma_H, sigma_T=0):
    x = add_pose_noise(x, sigma_H)
    if x.shape[0] == 12:
        x = add_t_noise(x, sigma_T)
    return x


def approx_instant_twist(H, dt=1.0):
    batch = H.shape[:-2]
    n_step = H.shape[-2]
    T = pp.se3(torch.empty((*batch, n_step, 6), dtype=H.dtype))
    # Assume no acceleration in the first interval
    T[..., 0, :] = pp.Log(H[..., 1, :] @ H[..., 0, :].Inv()) / dt

    for i in range(1, n_step):
        H_inter = pp.Exp(pp.se3(T[..., i-1, :] * dt / 2)) @ H[..., i-1, :]
        T[..., i, :] = -2 / dt * pp.Log(H_inter @ H[..., i, :].Inv())
    return T


def reward_by_zone(H, zones: typing.List[Zone], dist_scale=None):
    rewards = torch.zeros(H.shape[:-1], device=H.device, dtype=H.dtype)
    for zone in zones:
        if dist_scale is None:
            mask = ((H[..., 0] > zone.xmin) & (H[..., 0] < zone.xmax) &
                    (H[..., 1] > zone.ymin) & (H[..., 1] < zone.ymax) &
                    (H[..., 2] > zone.zmin) & (H[..., 2] < zone.zmax))
            rewards[mask] = torch.minimum(rewards[mask], torch.tensor([-1.0]))
            continue

        dx = torch.clip(torch.maximum(zone.xmin - H[..., 0], H[..., 0] - zone.xmax), min=0)
        dy = torch.clip(torch.maximum(zone.ymin - H[..., 1], H[..., 1] - zone.ymax), min=0)
        dz = torch.clip(torch.maximum(zone.zmin - H[..., 2], H[..., 2] - zone.zmax), min=0)
        dist = torch.sqrt(dx*dx + dy*dy + dz*dz)
        rewards = torch.minimum(rewards, -torch.exp(-4.6*dist/dist_scale))
    return rewards


def reward_distance_to_end(H, xmin, xmax):
    return -(torch.linalg.vector_norm((H[..., :3] - H[..., -1, :3][..., None, :]), dim=-1)/np.linalg.norm(xmax-xmin))**2


class BSplineDefault:
    method = 'bspline'
    mode = 'random'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = 0.08  # s
    sigma_H = None
    sigma_T = None
    zones = []
    zone_dist_scale = None
    dist_reward = False
    reward_weights = None


class BSplinePoseOnly:
    method = 'bspline'
    mode = 'random'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = []
    zone_dist_scale = None
    dist_reward = False
    reward_weights = None


class BSplineNoisyPoseOnly:
    method = 'bspline'
    mode = 'random'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = 1e-5
    sigma_T = None
    zones = []
    zone_dist_scale = None
    dist_reward = False
    reward_weights = None


class BSplinePoseObstacle:
    method = 'bspline'
    mode = 'random'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = [Zone(xmin=0.45, ymin=0.45, zmin=0, xmax=0.55, ymax=0.55, zmax=1)]
    zone_dist_scale = None
    dist_reward = False
    reward_weights = np.array([5])


class BSplinePoseDist:
    method = 'bspline'
    mode = 'random'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = []
    zone_dist_scale = None
    dist_reward = True
    reward_weights = None


class BSplinePoseMixReward:
    method = 'bspline'
    mode = 'random'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = [Zone(xmin=0.45, ymin=0.45, zmin=0, xmax=0.55, ymax=0.55, zmax=1)]
    zone_dist_scale = None
    dist_reward = True
    reward_weights = np.array([5, 1])


class LinesPoseObst:
    method = 'chspline'
    mode = 'lines'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = [Zone(xmin=0.4, ymin=0.4, zmin=0, xmax=0.6, ymax=0.6, zmax=1)]
    zone_dist_scale = None
    dist_reward = False
    reward_weights = None


class LinesPoseDist:
    method = 'chspline'
    mode = 'lines'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = []
    zone_dist_scale = None
    dist_reward = True
    reward_weights = None


class LinesPoseMixRew:
    method = 'chspline'
    mode = 'lines'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = [Zone(xmin=0.4, ymin=0.4, zmin=0, xmax=0.6, ymax=0.6, zmax=1)]
    zone_dist_scale = None
    dist_reward = True
    reward_weights = np.array([5, 1])


class LinesPoseObstDist:
    method = 'chspline'
    mode = 'lines'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = None
    sigma_T = None
    zones = [Zone(xmin=0.4, ymin=0.4, zmin=0, xmax=0.6, ymax=0.6, zmax=1)]
    zone_dist_scale = None
    dist_reward = True
    reward_weights = None


class BSplineTesting:
    method = 'bspline'
    mode = 'lines'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 1000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s
    sigma_H = 0.1
    sigma_T = None
    zones = [Zone(xmin=0, ymin=0, zmin=0, xmax=0.5, ymax=0.5, zmax=0.5), Zone(xmin=0, ymin=0, zmin=0, xmax=0.4, ymax=0.4, zmax=0.4)]
    zone_dist_scale = 0.5
    dist_reward = True
    reward_weights = np.array([100, 1])


class SplineGenerator(object):
    def __init__(self, xmin=np.array([0, 0, 0]), xmax=np.array([1, 1, 1]), method='b'):
        assert (xmin < xmax).all()
        self.xmin = xmin
        self.xmax = xmax
        self.interpolator = pp.chspline if method == 'ch' else pp.bspline

    def generate_from_support(self, support: pp.SE3_type, interval: float = 0.1):
        assert ((self.xmin <= support.tensor()[:, :, :3].numpy())
                & (support.tensor()[:, :, :3].numpy() <= self.xmax)).all()
        H = pp.SE3(self.interpolator(support, interval=interval))
        return H

    @torch.no_grad()
    def generate_random(self, n_traj=1, n_step=10, n_interval=1):
        assert n_interval >= 1
        n_support = n_interval + 1 if self.interpolator is pp.chspline else n_interval + 3
        interval = 1/n_step

        support_x = torch.rand((n_traj, n_support, 3)) * (self.xmax - self.xmin) + self.xmin
        # Generate random Gaussian vector to sample uniform directions, scale rotation uniformly [0, 2pi]
        support_r = torch.randn((n_traj, n_support, 3))
        support_r = pp.so3(2 * torch.pi * torch.rand((n_traj, n_support, 1)) *
                           support_r / torch.norm(support_r, dim=2, keepdim=True)).Exp()
        support = pp.SE3(torch.cat((support_x, support_r.tensor()), dim=2))
        return self.generate_from_support(support, interval=interval), (support, interval)

    @torch.no_grad()
    def generate_lines(self, n_traj=1, n_step=10, n_interval=1):
        # Rotation interpolated with chspline, since bspline is not supported for SO(3)
        n_support = n_interval + 1
        n_points = n_step*n_interval + 1
        end_points = torch.rand((n_traj, 2, 3)) * (self.xmax - self.xmin) + self.xmin
        diff = end_points[:, 1:2] - end_points[:, 0:1]

        # Assume progression along the line is cubic in time: y=-2t^3+3t^2
        t = np.linspace(0, 1, n_points)
        progression = -2*t**3+3*t**2
        x = end_points[:, 0:1] + diff.repeat((1, n_points, 1)) * progression[:, None]

        # Rotation interpolated with chspline, since bspline is not supported for SO(3)
        support_r = torch.randn((n_traj, n_support, 3))
        support_r = pp.so3(2 * torch.pi * torch.rand((n_traj, n_support, 1)) *
                           support_r / torch.norm(support_r, dim=2, keepdim=True)).Exp()
        r = pp.SO3(pp.chspline(support_r, 1/n_step))
        return pp.SE3(torch.cat([x, r.tensor()], dim=-1)), (support_r, 1/n_step)


class SplineDataset(object):
    def __init__(self, config, repres='se3'):
        assert repres in {'cart', 'se3'}
        self.config = config
        self.repres = repres
        self.splines = SplineGenerator(config.xmin, config.xmax, config.method)
        self.mode = config.mode
        self.nr_trajectories = config.nr_trajectories
        self.nr_intervals = config.nr_intervals   # nr interpolated segments in a trajectory
        self.nr_steps = config.nr_steps       # interpolation steps per trajectory segment
        self.dt = config.dt  # s
        self.sigma_H = config.sigma_H
        self.sigma_T = config.sigma_T
        self.obstacles = config.zones
        self.obst_dist_scale = config.zone_dist_scale
        self.dist_reward = config.dist_reward
        self.reward_weights = config.reward_weights
        self.data = self.generate()

    def generate(self):
        if self.mode == 'random':
            path, _ = self.splines.generate_random(n_traj=self.nr_trajectories,
                                                   n_step=self.nr_steps, n_interval=self.nr_intervals)
        elif self.mode == 'lines':
            path, _ = self.splines.generate_lines(n_traj=self.nr_trajectories,
                                                  n_step=self.nr_steps, n_interval=self.nr_intervals)

        observations = path.Log().tensor() if self.repres == 'se3' else path.tensor()

        if self.dt is not None:
            twist = approx_instant_twist(path, dt=self.dt)
            observations = torch.cat([observations, twist.tensor()], dim=-1)

        if self.sigma_H is not None or self.sigma_T is not None:
            observations = add_noise(observations, self.sigma_H, self.sigma_T)

        rewards = self.generate_rewards(path.tensor(), repres='cart')

        terminals = torch.zeros((*path.shape[:-1], 1), device=path.device, dtype=torch.bool)
        terminals[:, -1] = 1
        timeouts = torch.zeros((*path.shape[:-1], 1), device=path.device, dtype=torch.bool)
        actions = torch.zeros((*path.shape[:-1], 1), device=path.device)
        return {'observations': observations, 'actions': actions, 'rewards': rewards,
                'terminals': terminals, 'timeouts': timeouts}

    def generate_rewards(self, paths: torch.Tensor, repres='se3'):
        if repres == 'se3':
            paths = pp.se3(paths).Exp().tensor()

        rewards = []
        if len(self.obstacles) > 0:
            rewards.append(reward_by_zone(paths, self.obstacles, dist_scale=self.obst_dist_scale))
        if self.dist_reward:
            rewards.append(reward_distance_to_end(paths, self.config.xmin, self.config.xmax))

        if len(rewards) == 0:
            rewards = torch.zeros((*paths.shape[:-1], 1), device=paths.device)
        else:
            rewards = torch.stack(rewards, dim=-1)
        if self.reward_weights is not None:
            rewards = torch.sum(rewards * self.reward_weights, dim=-1, keepdim=True)
        return rewards

    def __call__(self, *args, **kwargs):
        for i in range(self.data['observations'].shape[0]):
            episode = {}
            for k in self.data.keys():
                episode[k] = np.array(self.data[k][i])
            yield episode


if __name__ == "__main__":
    # config = BSplineTesting
    # bins = np.linspace(0, 20, 100)
    # sigmas = [100, 1, 0.5, 1e-3, 1e-4, 1e-5, 0]#, 1e-5, 0]
    # scores = []
    # plt.figure()
    # for sigma in sigmas:
    #     config.sigma_H = sigma
    #     dataset = SplineDataset(config)
    #     scores.append(kinematic_pose_consistency(dataset.data['observations'], norm=True).numpy().flatten())
    #
    # lines = plt.hist(scores, bins=bins, stacked=False, histtype='step', density=True,
    #                  label=[f'sigmaH: {sigma}, mean: {score.mean():.2f}' for sigma, score in zip(sigmas, scores)])
    # plt.legend()
    # plt.xlabel('Kinematic pose score, normalized')
    # plt.ylabel('Density')
    # plt.show()

    gen = SplineGenerator(xmin=np.array([0, 0, 0]), xmax=np.array([1, 1, 1]), method='bs')
    paths, (supp, interval) = gen.generate_random(n_traj=2, n_step=10, n_interval=3)
    # delta_t = 0.1
    # twists = approx_instant_twist(paths, delta_t)
    # x = torch.cat([paths.Log(), twists], dim=-1)
    # res = kinematic_consistency(x, delta_t)
    # res_norm = kinematic_consistency(x, delta_t, norm=True)
    # dist_SE3(paths[..., :-1, :], paths[..., 1:, :])
    # torch.mean(dist_SE3(pp.Exp(twists[..., :-1, :]) @ paths[..., :-1, :], paths[..., 1:, :]))
    # fig, ax = plot_trajectory(paths, show=False, plot_end=True, detail_ends=4, step=2)

    # config = BSplinePoseMixReward
    # dataset = SplineDataset(config)

    #
    config = BSplineTesting
    dataset = SplineDataset(config)
    gen = SplineGenerator(xmin=np.array([0, 0, 0]), xmax=np.array([1, 1, 1]), method='bs')
    paths, (supp, interval) = gen.generate_lines(n_traj=1, n_step=10, n_interval=3)
    fig, ax = plot_trajectory(paths, show=True, as_equal=True, block=False)
    ax.set_title('Paths')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    fig.show()
