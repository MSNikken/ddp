import numpy as np
import torch, pypose as pp

from diffuser.utils.visualization import plot_trajectory
from diffuser.models.helpers import dist_SE3, kinematic_consistency


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


class BSplineDefault:
    method = 'bspline'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = 0.08  # s


class BSplinePoseOnly:
    method = 'bspline'
    xmin = np.array([0, 0, 0])
    xmax = np.array([1, 1, 1])
    nr_trajectories = 10000
    nr_intervals = 4  # nr interpolated segments in a trajectory
    nr_steps = 50  # interpolation steps per trajectory segment
    dt = None  # s


class BSplineTesting:
    method = 'bspline'
    xmin = np.array([1, 1, 1])*0
    xmax = np.array([1, 1, 1])*4
    nr_trajectories = 1000
    nr_intervals = 2  # nr interpolated segments in a trajectory
    nr_steps = 10  # interpolation steps per trajectory segment
    dt = 0.08  # s


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


class SplineDataset(object):
    def __init__(self, config, repres='se3'):
        assert repres in {'cart', 'se3'}
        self.repres = repres
        self.splines = SplineGenerator(config.xmin, config.xmax, config.method)
        self.nr_trajectories = config.nr_trajectories
        self.nr_intervals = config.nr_intervals   # nr interpolated segments in a trajectory
        self.nr_steps = config.nr_steps       # interpolation steps per trajectory segment
        self.dt = config.dt  # s
        self.data = self.generate()

    def generate(self):
        path, _ = self.splines.generate_random(n_traj=self.nr_trajectories, n_step=self.nr_steps, n_interval=self.nr_intervals)
        if self.repres == 'se3':
            path = pp.Log(path)
        if self.dt is not None:
            twist = approx_instant_twist(path, dt=self.dt)
            observations = torch.cat([path.tensor(), twist.tensor()], dim=-1)
        else:
            observations = path.tensor()
        rewards = torch.zeros((*path.shape[:-1], 1), device=path.device)
        terminals = torch.zeros((*path.shape[:-1], 1), device=path.device, dtype=torch.bool)
        terminals[:, -1] = 1
        timeouts = torch.zeros((*path.shape[:-1], 1), device=path.device, dtype=torch.bool)
        actions = torch.zeros((*path.shape[:-1], 1), device=path.device)
        return {'observations': observations, 'actions': actions, 'rewards': rewards,
                'terminals': terminals, 'timeouts': timeouts}

    def __call__(self, *args, **kwargs):
        for i in range(self.data['observations'].shape[0]):
            episode = {}
            for k in self.data.keys():
                episode[k] = np.array(self.data[k][i])
            yield episode


if __name__ == "__main__":
    gen = SplineGenerator(xmin=np.array([0, 0, 0]), xmax=np.array([1, 1, 1]), method='bs')
    paths, (supp, interval) = gen.generate_random(n_traj=2, n_step=10, n_interval=3)
    delta_t = 0.1
    twists = approx_instant_twist(paths, delta_t)
    x = torch.cat([paths.Log(), twists], dim=-1)
    res = kinematic_consistency(x, delta_t)
    res_norm = kinematic_consistency(x, delta_t, norm=True)
    dist_SE3(paths[..., :-1, :], paths[..., 1:, :])
    torch.mean(dist_SE3(pp.Exp(twists[..., :-1, :]) @ paths[..., :-1, :], paths[..., 1:, :]))
    fig, ax = plot_trajectory(paths, show=False)
    ax.set_title('B spline')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()

    gen.interpolator = pp.chspline
    paths = gen.generate_from_support(supp, interval)
    twists = approx_instant_twist(paths)
    fig, ax = plot_trajectory(paths, show=False)
    ax.set_title('CH spline')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()
    pass
