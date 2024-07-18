import time
import warnings

import numpy as np
import torch


class BaseDiffusionPlanner(object):
    def __init__(self, horizon, dt_plan, state_dim, action_dim, device):
        self.device = device
        self.horizon = horizon
        self.dt_plan = dt_plan
        self.plan = torch.empty((horizon, state_dim), device=device)
        self.action = torch.empty((horizon, action_dim), device=device)
        self.plan_starttime = 0
        self.obs_indices = []

    def add_observation(self, obs):
        obs_index = self.now_index()
        if obs_index >= self.horizon:
            warnings.warn('Observation outside of planning horizon.')
            return
        self.plan[obs_index] = obs
        self.obs_indices.append(obs_index)

    def generate_plan(self, start, end):
        self.plan_starttime = time.time_ns()
        self.obs_indices = [-1]
        self.plan[-1] = end
        self.add_observation(start)
        self.update_plan()

    def update_plan(self):
        raise NotImplementedError

    def next_setpoint(self):
        action_index = self.now_index()
        if action_index >= self.horizon:
            warnings.warn('No planned action')
            return None
        return self.action[action_index], action_index == self.horizon-1

    def now_index(self):
        return np.floor((time.time_ns() - self.plan_starttime) / (self.dt_plan * 1e9))


class GausInvDynPlanner(BaseDiffusionPlanner):
    def __init__(self, model, horizon, dt_plan, dt_sample, state_dim, action_dim, device, mode='pos'):
        super().__init__(horizon, dt_plan, state_dim, action_dim, device)
        if mode not in ['pos', 'vel']:
            raise AttributeError('Invalid action mode.')
        self.mode = mode
        self.model = model
        self.dt_sample = dt_sample
        self.sample_step_index = np.round(dt_sample / dt_plan)

    def update_plan(self):
        conditions = {i: self.plan[i] for i in self.obs_indices}
        self.plan = self.model(conditions, horizon=self.horizon)
        now = self.now_index()
        if now >= self.horizon:
            warnings.warn('No planned action')
            return
        if self.mode == 'pos':
            self.action[now:-self.sample_step_index] = self.plan[now + 1:, :7]
            self.action[-self.sample_step_index:] = self.plan[-1, :7]
        else:
            self.action[now:-self.sample_step_index] = self.plan[now + 1:, 7:13]
            self.action[-self.sample_step_index:] = self.plan[-1, :7]
