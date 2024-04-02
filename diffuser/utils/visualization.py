import matplotlib.pyplot as plt
import torch
import pypose as pp


def _plot_position(ax, traj: pp.SE3_type, indices, marker=True):
    traj = traj.numpy()
    x = traj[indices, 0]
    y = traj[indices, 1]
    z = traj[indices, 2]
    if marker:
        ax.scatter(x, y, z, marker='o')
    ax.plot(x, y, z, zorder=0.5)
    ax.scatter(x[0], y[0], z[0], marker='*', s=100)


def _plot_orientation(ax, H: pp.SE3_type, indices, scale=0.05):
    nr_frames = int((indices.stop - indices.start) / indices.step)
    origin = torch.zeros((nr_frames, 3), dtype=torch.double)
    x_vec, y_vec, z_vec = (torch.zeros((nr_frames, 3), dtype=torch.double),
                           torch.zeros((nr_frames, 3), dtype=torch.double),
                           torch.zeros((nr_frames, 3), dtype=torch.double))
    x_vec[:, 0] = scale
    y_vec[:, 1] = scale
    z_vec[:, 2] = scale

    origin = H @ pp.cart2homo(origin)
    x_vec = H @ pp.cart2homo(x_vec)
    y_vec = H @ pp.cart2homo(y_vec)
    z_vec = H @ pp.cart2homo(z_vec)

    for i in range(nr_frames):
        ax.plot([origin[i, 0], x_vec[i, 0]], [origin[i, 1], x_vec[i, 1]], [origin[i, 2], x_vec[i, 2]], c='r')
        ax.plot([origin[i, 0], y_vec[i, 0]], [origin[i, 1], y_vec[i, 1]], [origin[i, 2], y_vec[i, 2]], c='g')
        ax.plot([origin[i, 0], z_vec[i, 0]], [origin[i, 1], z_vec[i, 1]], [origin[i, 2], z_vec[i, 2]], c='b')


def plot_trajectory(traj, step=1, show=True, block=True, marker=False, rot=True):
    traj = traj.cpu()
    if traj.ndim == 2:
        traj = traj[None, ...]
    if rot and not type(traj) == pp.SE3_type:
        traj = pp.SE3(traj)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    indices = slice(0, traj.shape[1], step)
    for tau in traj:
        _plot_position(ax, tau, indices, marker)
        if rot:
            _plot_orientation(ax, tau, indices)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show:
        plt.show(block=block)
    return plt, ax
