import pypose as pp
import torch


def traj_euc2se3(x: torch.Tensor):
    traj_pos = x[..., :, :3]
    traj_rot = x[..., :, 3:7]
    traj_rot = torch.nn.functional.normalize(traj_rot, dim=-1)
    traj_pose = pp.SE3(torch.cat([traj_pos, traj_rot], dim=-1)).Log()
    traj_twist = x[..., :, 7:13]
    return torch.cat([traj_pose.tensor(), traj_twist], dim=-1)
