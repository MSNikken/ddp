# import gym
# import d4rl
#
# env = gym.make('kitchen-complete-v0')
#
# # Automatically download and return the dataset
# dataset = env.get_dataset()
# print(dataset['observations'].shape) # An (N, dim_observation)-dimensional numpy array of observations
# print(dataset['actions'].shape) # An (N, dim_action)-dimensional numpy array of actions
# print(dataset['rewards'].shape) # An (N,)-dimensional numpy array of rewards
#
# action = env.action_space.sample()
# env.reset()
# #obs, rew, term, trunc, info, done = env.step(action)
# env.step(action)


import mujoco
import os
# xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'environments/franka/franka_panda.xml')
# mj_model = mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(mj_model)
# print(data.geom(1).xpos)
#
# mujoco.mj_kinematics(mj_model, data)
# print(data.geom(1).xpos)

# bodies:
# panda0_link7
# panda0_leftfinger

# import torch
# import pypose as pp
# import numpy as np
#
# H1 = pp.identity_SE3()
# H2 = pp.SE3([10, 20, 30, 0, 0, np.sqrt(1/2), np.sqrt(1/2)])
# H3 = pp.SE3([5, 6, 7, 0, 0, 0, 1])
# H4 = pp.SE3([0, 0, 0, 0, 0, np.sqrt(1/2), np.sqrt(1/2)])
# T1 = pp.identity_se3()
# T2 = pp.se3([2, 3, 4, 0, 0, 0])
# T3 = pp.se3([1, 0, 0, 0, 0, 0])
# T4 = pp.se3([1, 0, 0, 1, 0, 0])
#
# from diffuser.datasets.preprocessing import joints_to_cart, cart_to_se3, only_trajectory
# import gym
# import d4rl
#
# from diffuser.models.helpers import kinematic_consistency
#
# env = gym.make('kitchen-complete-v0')
# dt = env.env.dt
# dataset = env.get_dataset()
# fn0 = only_trajectory(None)
# fn1 = joints_to_cart(None)
# fn2 = cart_to_se3(None)
# dataset = fn2(fn1(fn0(dataset)))
# indices = np.where(dataset['terminals'])[0]
# traj = dataset['observations'][:indices[0], :]
# kinematic_consistency(traj, dt)
# pass


# import necessary libraries
import torch

from diffuser.datasets.normalization import GaussianNormalizer, CDFNormalizer

# define a tensor
data = torch.tensor([5., 1, 2, 3, 3, 3], requires_grad=False)

normalizer = GaussianNormalizer(data.numpy())
#normalizer = CDFNormalizer(data.numpy())
sample = torch.tensor(2.5, requires_grad=False)
norm_sample = normalizer(sample)
theta = torch.tensor(3., requires_grad=True)

# define a function using above defined
# tensor

x = theta * norm_sample
loss = 2*normalizer.unnormalize(x)
loss.backward()
print("theta.grad:", theta.grad)

x = theta * norm_sample
loss = 2*x
loss.backward()
print("theta.grad:", theta.grad)

x = theta * norm_sample
loss = 2*x
loss.backward()
print("theta.grad:", theta.grad)
