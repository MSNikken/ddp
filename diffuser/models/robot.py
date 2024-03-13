import numpy as np
import torch
import pypose as pp
import mujoco


class Robot(object):
    def __init__(self, xml_path, end_effector=None):
        self.mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data: mujoco.MjData = mujoco.MjData(self.mj_model)

        self.njoints = self.mj_data.qpos.shape[0]
        self.end_effector = end_effector or self.mj_model.nbody - 1

    def fwd_kinematics(self, qpos):
        self.mj_data.qpos = qpos
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return self.mj_data.xpos, self.mj_data.xquat

    def fwd_diff_kinematics(self, qpos, qvel):
        jacp, jacr = self.body_jacobian(qpos)
        lin_vel = jacp @ qvel
        rot_vel = jacr @ qvel
        return lin_vel, rot_vel

    def body_jacobian(self, qpos):
        # Note: this is the jacobian with respect to the body, which may not be the desired end effector frame

        self.mj_data.qpos = qpos

        jacp = np.empty((3, self.mj_model.nv))
        jacr = np.empty((3, self.mj_model.nv))

        # Required before jacobian calculation:
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)

        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, self.end_effector)
        return jacp, jacr
