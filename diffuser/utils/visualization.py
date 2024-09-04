import matplotlib.pyplot as plt
import numpy as np
import torch
import pypose as pp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

view_dict = {'x': 0, 'y': 1, 'z': 2}


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


def plot_trajectory(traj, step=1, show=True, block=True, marker=False, rot=True, plot_end=False, detail_ends=1,
                    as_equal=False):
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

    if as_equal:
        data_length = (traj.tensor()[..., :3].view(-1, 3).max(dim=0).values -
                    traj.tensor()[..., :3].view(-1, 3).min(dim=0).values)
        data_half = traj.tensor()[..., :3].view(-1, 3).min(dim=0).values + data_length/2
        half_ax_length = 1.03 * data_length.max()/2
        ax.set_xlim(data_half[0] - half_ax_length, data_half[0] + half_ax_length)
        ax.set_ylim(data_half[1] - half_ax_length, data_half[1] + half_ax_length)
        ax.set_zlim(data_half[2] - half_ax_length, data_half[2] + half_ax_length)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show:
        plt.show(block=block)
    return fig, ax


def _plot_position_2d(ax, traj: pp.SE3_type, dim1, dim2, indices, marker=True, line=True, start_end_marker=True,
                      skip_final_line=False, line_kwargs=None, marker_kwargs=None):
    if marker_kwargs is None:
        marker_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    traj = traj.numpy()
    if line:
        line_indices = indices[:-1] if skip_final_line else indices
        ax.plot(traj[line_indices, dim1], traj[line_indices, dim2], zorder=0.5, **line_kwargs)
    if marker:
        ax.scatter(traj[indices, dim1], traj[indices, dim2], marker='^', zorder=0, s=20, **marker_kwargs)
    if start_end_marker:
        ax.scatter(traj[0, dim1], traj[0, dim2], marker='*', color='tab:orange', s=100, **marker_kwargs)
        ax.scatter(traj[-1, dim1], traj[-1, dim2], marker='D', color='tab:green', s=50, **marker_kwargs)


def _plot_orientation_2d(ax, H: pp.SE3_type, dim1, dim2, indices, scale=0.05):
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
        ax.plot([origin[i, dim1], x_vec[i, dim1]], [origin[i, dim2], x_vec[i, dim2]], c='r')
        ax.plot([origin[i, dim1], y_vec[i, dim1]], [origin[i, dim2], y_vec[i, dim2]], c='g')
        ax.plot([origin[i, dim1], z_vec[i, dim1]], [origin[i, dim2], z_vec[i, dim2]], c='b')


def plot_trajectory_2d(traj, step=1, show=True, block=True, marker=False, rot=True, plot_end=False, detail_ends=1,
                    as_equal=False, view='xy', fig=None, ax=None):
    plot_dims = [view_dict[dim] for dim in [*view]]
    traj = tensor2batch_traj(traj)

    if rot:
        traj = normalize_quaternions(traj)
        scale = torch.max(traj.tensor()[..., :3].view(-1, 3).max(dim=0).values -
                          traj.tensor()[..., :3].view(-1, 3).min(dim=0).values) * 0.05

    indices = np.arange(0, traj.shape[1], step)
    if plot_end and not np.mod(traj.shape[1], step) == 1:
        indices = np.append(indices, traj.shape[1]-1)

    fig = plt.figure() if fig is None else fig
    ax = fig.add_subplot() if ax is None else ax

    if detail_ends > 1:
        left_tail = np.arange(0, np.minimum(detail_ends, traj.shape[1]))
        right_tail = np.arange(np.maximum(left_tail[-1]+1, traj.shape[1] - detail_ends), traj.shape[1])
        tail_indices = np.concatenate([left_tail, right_tail])
        plt.gca().set_prop_cycle(None)  # Reset colors to match positions
        for tau in traj:
            _plot_position_2d(ax, tau, *plot_dims, tail_indices, marker=True, line=False, start_end_marker=False)

    plt.gca().set_prop_cycle(None)  # Reset colors to match positions
    for tau in traj:
        _plot_position_2d(ax, tau, *plot_dims, indices, marker)
        if rot:
            _plot_orientation_2d(ax, tau, *plot_dims, indices, scale=scale)

    if as_equal:
        data_length = (traj.tensor()[..., :3].view(-1, 3).max(dim=0).values -
                    traj.tensor()[..., :3].view(-1, 3).min(dim=0).values)
        data_half = traj.tensor()[..., :3].view(-1, 3).min(dim=0).values + data_length/2
        half_ax_length = 1.03 * data_length.max()/2
        ax.set_xlim(data_half[plot_dims[0]] - half_ax_length, data_half[plot_dims[0]] + half_ax_length)
        ax.set_ylim(data_half[plot_dims[1]] - half_ax_length, data_half[plot_dims[1]] + half_ax_length)

    ax.set_xlabel(view[0])
    ax.set_ylabel(view[1])

    if show:
        plt.show(block=block)
    return fig, ax


def tensor2batch_traj(traj):
    traj = torch.tensor(traj) if isinstance(traj, np.ndarray) else traj.cpu()
    if traj.ndim == 2:
        traj = traj[None, ...]
    if not (isinstance(traj, pp.LieTensor) and traj.ltype == pp.SE3_type):
        if traj.shape[-1] == 7:
            traj = pp.SE3(traj)
        else:
            traj = pp.Exp(pp.se3(traj))
    return traj


def plot_trajectory_summary_2d(traj, view='xy', show=True, block=False, as_equal=False):
    plot_dims = [view_dict[dim] for dim in [*view]]
    traj = tensor2batch_traj(traj)

    fig = plt.figure()
    ax = fig.add_subplot()

    for tau in traj:
        _plot_position_2d(ax, tau, *plot_dims, range(traj.shape[1]), marker=False, start_end_marker=False,
                          skip_final_line=True, line_kwargs={'color': '0.4'})
    mean_pos = traj.mean(dim=0)
    _plot_position_2d(ax, mean_pos, *plot_dims, range(traj.shape[1]), marker=False, skip_final_line=True,
                      line_kwargs={'linewidth': 2.0})

    if as_equal:
        data_length = (traj.tensor()[..., :3].view(-1, 3).max(dim=0).values -
                    traj.tensor()[..., :3].view(-1, 3).min(dim=0).values)
        data_half = traj.tensor()[..., :3].view(-1, 3).min(dim=0).values + data_length/2
        half_ax_length = 1.03 * data_length.max()/2
        ax.set_xlim(data_half[plot_dims[0]] - half_ax_length, data_half[plot_dims[0]] + half_ax_length)
        ax.set_ylim(data_half[plot_dims[1]] - half_ax_length, data_half[plot_dims[1]] + half_ax_length)

    ax.set_xlabel(view[0])
    ax.set_ylabel(view[1])

    if show:
        plt.show(block=block)
    return fig, ax


def draw_rectangles(ax, rects, facecolor='r', edgecolor='none', alpha=0.5):
    rectangles = [Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1]) for rect in rects]
    pc = PatchCollection(rectangles, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_collection(pc)
