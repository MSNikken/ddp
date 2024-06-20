import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import pypose as pp


def normalize_quaternions(batch_SE3: pp.SE3_type):
    batch_SE3.tensor()[..., 3:] = batch_SE3.tensor()[..., 3:] / torch.linalg.norm(batch_SE3.tensor()[..., 3:], axis=-1, keepdims=True)
    return batch_SE3


def _plot_position(ax, traj: pp.SE3_type, indices, marker=True, line=True, start_marker=True):
    traj = traj.numpy()
    x = traj[indices, 0]
    y = traj[indices, 1]
    z = traj[indices, 2]
    if marker:
        ax.scatter(x, y, z, marker='^', depthshade=False, zorder=0, s=20)
    if line:
        ax.plot(x, y, z, zorder=0.5)
    if start_marker:
        ax.scatter(x[0], y[0], z[0], marker='*', s=100)


def _plot_orientation(ax, H: pp.SE3_type, indices, scale=0.05):
    nr_frames = indices.size
    origin = torch.zeros((nr_frames, 3), dtype=H.dtype)
    x_vec, y_vec, z_vec = (torch.zeros((nr_frames, 3), dtype=H.dtype),
                           torch.zeros((nr_frames, 3), dtype=H.dtype),
                           torch.zeros((nr_frames, 3), dtype=H.dtype))

    x_vec[:, 0] = scale
    y_vec[:, 1] = scale
    z_vec[:, 2] = scale

    origin = H[indices] @ pp.cart2homo(origin)
    x_vec = H[indices] @ pp.cart2homo(x_vec)
    y_vec = H[indices] @ pp.cart2homo(y_vec)
    z_vec = H[indices] @ pp.cart2homo(z_vec)

    for i in range(nr_frames):
        ax.plot([origin[i, 0], x_vec[i, 0]], [origin[i, 1], x_vec[i, 1]], [origin[i, 2], x_vec[i, 2]], c='r')
        ax.plot([origin[i, 0], y_vec[i, 0]], [origin[i, 1], y_vec[i, 1]], [origin[i, 2], y_vec[i, 2]], c='g')
        ax.plot([origin[i, 0], z_vec[i, 0]], [origin[i, 1], z_vec[i, 1]], [origin[i, 2], z_vec[i, 2]], c='b')


def plot_trajectory(traj, step=1, show=True, block=True, marker=False, rot=True, plot_end=False, detail_ends=1):
    traj = torch.tensor(traj) if isinstance(traj, np.ndarray) else traj.cpu()
    if traj.ndim == 2:
        traj = traj[None, ...]

    if not (isinstance(traj, pp.LieTensor) and traj.ltype == pp.SE3_type):
        if traj.shape[-1] == 7:
            traj = pp.SE3(traj)
        else:
            traj = pp.Exp(pp.se3(traj))

    if rot:
        traj = normalize_quaternions(traj)
        scale = torch.max(traj.tensor()[..., :3].view(-1, 3).max(dim=0).values -
                          traj.tensor()[..., :3].view(-1, 3).min(dim=0).values) * 0.05

    indices = np.arange(0, traj.shape[1], step)
    if plot_end and not np.mod(traj.shape[1], step) == 1:
        indices = np.append(indices, traj.shape[1]-1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if detail_ends > 1:
        left_tail = np.arange(0, np.minimum(detail_ends, traj.shape[1]))
        right_tail = np.arange(np.maximum(left_tail[-1]+1, traj.shape[1] - detail_ends), traj.shape[1])
        tail_indices = np.concatenate([left_tail, right_tail])
        plt.gca().set_prop_cycle(None)  # Reset colors to match positions
        for tau in traj:
            _plot_position(ax, tau, tail_indices, marker=True, line=False, start_marker=False)

    plt.gca().set_prop_cycle(None)  # Reset colors to match positions
    for tau in traj:
        _plot_position(ax, tau, indices, marker)
        if rot:
            _plot_orientation(ax, tau, indices, scale=scale)



    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show:
        plt.show(block=block)
    return fig, ax
