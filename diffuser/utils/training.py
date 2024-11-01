import os
import copy
import numpy as np
import torch
import einops
import pdb

import wandb

import diffuser
from copy import deepcopy

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from ..models.helpers import kinematic_pose_consistency, traj_euc2se3


#from ml_logger import logger

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        dataset_val=None,
        val_batch_size=32,
        val_nr_batch=2,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        inference_returns=1,
        inference_horizon=100,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataset_val = dataset_val
        self.val_nr_batch = val_nr_batch

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_val = cycle(torch.utils.data.DataLoader(
            self.dataset_val, batch_size=val_batch_size, num_workers=0, shuffle=True, pin_memory=True
        )) if self.dataset_val is not None else None
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference
        self.inference_returns = np.array(inference_returns, dtype=np.float32)
        self.inference_horizon = inference_horizon

        self.reset_parameters()
        self.step = 0

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        wandb.watch(self.model.model, log="all", log_freq=self.log_freq)

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'step: {self.step}| loss: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k: v.detach().item() for k, v in infos.items()}
                metrics['loss'] = loss.detach().item()

                # Note: loss is divided by gradient_accumulate_every
                # Postpone committing to sampling step
                commit = not (self.sample_freq and self.step % self.sample_freq == 0)
                wandb.log({**metrics, 'step': self.step}, commit=commit)

            if self.step == 0 and self.sample_freq:
                log_entry = self.render_reference(self.n_reference)
                if log_entry is not None:
                    wandb.log(log_entry, commit=False)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if (self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion
                        or self.model.__class__ == diffuser.models.lie_diffusion.SE3Diffusion):
                    logs = [self.validate(), self.inv_render_samples(), self.render_inpainting_samples(),
                            self.render_scenario_samples(), self.kinematic_validation()]
                    log_dict = {}
                    logs = [log for log in logs if log is not None]
                    for log in logs:
                        log_dict.update(log)
                    wandb.log(log_dict)
                elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
                    pass
                else:
                    self.render_samples()

            self.step += 1

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        wandb.save(savepath, policy="now")
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        return self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, self.dataset.returns_dim)*self.inference_returns, self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns,
                                                                 horizon=self.inference_horizon)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns, horizon=self.inference_horizon)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # normed_observations = np.concatenate([
            #     np.repeat(normed_conditions, n_samples, axis=0),
            #     normed_observations
            # ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        log_entries = {}
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, self.dataset.returns_dim)*self.inference_returns, self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns,
                                                                 horizon=self.inference_horizon)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns, horizon=self.inference_horizon)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # normed_observations = np.concatenate([
            #     np.repeat(normed_conditions, n_samples, axis=0),
            #     normed_observations
            # ], axis=1)

            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            log_entry = self.renderer.composite(savepath, observations)
            if log_entry is not None:
                log_entries.update(log_entry)
        if len(log_entries) > 0:
            return log_entries

    def render_inpainting_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        log_entries = {}
        for i in range(batch_size):

            # get a two random points in normalized space
            conditions_sample = torch.rand(2, 1, self.dataset.observation_dim, device=self.device)*2 - 1
            conditions = {k: v for k, v in zip([0, self.inference_horizon-1], conditions_sample)}
            conditions = to_device(conditions, self.device)
            # repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            # [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, self.dataset.returns_dim)*self.inference_returns, self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns,
                                                                 horizon=self.inference_horizon)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns, horizon=self.inference_horizon)

            samples = to_np(samples)

            # [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(conditions_sample)[:, None]


            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join('images', f'inpainting_sample-{i}.png')
            log_entry = self.renderer.composite(savepath, observations)
            if log_entry is not None:
                log_entries.update(log_entry)
        if len(log_entries) > 0:
            return log_entries

    def render_scenario_samples(self, batch_size=2, n_samples=2):
        log_entries = {}
        # TODO: Configure scenario from config file
        scn_conditions = [torch.rand(3, 1, self.dataset.observation_dim, device=self.device) * 2 - 1,
                          torch.rand(3, 1, self.dataset.observation_dim, device=self.device) * 2 - 1]
        scn_conditions = [self.dataset.normalizer.unnormalize(cond, 'observations') for cond in scn_conditions]
        scn_conditions[0][0, :, :3] = torch.tensor([0.5, 0.35, 0.5])
        scn_conditions[0][1, :, :3] = torch.tensor([0.8, 0.5, 0.5])
        scn_conditions[0][2, :, :3] = torch.tensor([0.5, 0.65, 0.5])
        scn_conditions[1][0, :, :3] = torch.tensor([0.3, 0.5, 0.1])
        scn_conditions[1][1, :, :3] = torch.tensor([0.7, 0.5, 0.1])
        scn_conditions[1][2, :, :3] = torch.tensor([0.7, 0.5, 0.9])
        scn_conditions = [self.dataset.normalizer.normalize(cond, 'observations') for cond in scn_conditions]

        for i in range(batch_size):
            # get random points in normalized space
            conditions_sample = scn_conditions[i]
            conditions = {k: v for k, v in zip([0, int(self.inference_horizon/2), self.inference_horizon - 1],
                                               conditions_sample)}
            conditions = to_device(conditions, self.device)
            # repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            # [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, self.dataset.returns_dim) * self.inference_returns, self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns,
                                                                 horizon=self.inference_horizon)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns, horizon=self.inference_horizon)

            samples = to_np(samples)

            # [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join('images', f'scn_sample-{i}.png')
            log_entry = self.renderer.composite(savepath, observations)
            if log_entry is not None:
                log_entries.update(log_entry)
        if len(log_entries) > 0:
            return log_entries

    def kinematic_validation(self, batch_size=10, n_samples=10):
        paths = torch.empty((batch_size*n_samples, self.inference_horizon, 6), device=self.device)
        for i in range(batch_size):
            # get a two random points in normalized space
            conditions_sample = torch.rand(2, 1, self.dataset.observation_dim, device=self.device)*2 - 1
            conditions = {k: v for k, v in zip([0, self.inference_horizon-1], conditions_sample)}
            conditions = to_device(conditions, self.device)
            # repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            # [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, self.dataset.returns_dim)*self.inference_returns, self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns,
                                                                 horizon=self.inference_horizon)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns, horizon=self.inference_horizon)

            if self.model.__class__ == diffuser.models.lie_diffusion.SE3Diffusion:
                path = self.dataset.normalizer.unnormalize(samples, 'observations')[..., :6]
            else:
                path = traj_euc2se3(self.dataset.normalizer.unnormalize(samples, 'observations'), twist=False)

            paths[i*n_samples:i*n_samples+n_samples, ...] = path

        score = torch.flatten(kinematic_pose_consistency(paths, norm=True))
        k_top = 0.05
        top_k, _ = torch.topk(score, int(score.numel()*k_top))

        table = wandb.Table(data=[[s] for s in score], columns=['kin score'])
        table_topk = wandb.Table(data=[[s] for s in top_k], columns=['kin score'])

        #top = torch.topk(score, np.floor(batch_size*n_samples*0.05))
        #return {"validation/kin_mean": torch.mean(score), "validation/top5_percent": top[-1]}
        return {'validation/kin_score': wandb.plot.histogram(table, 'kin score',
                                                             title='Generation kin. score distribution'),
                f'validation/kin_score_top_{k_top}':
                    wandb.plot.histogram(table_topk, 'kin score',
                                         title=f'Top {k_top*100}% generation kin. score distribution')
                }

    def validate(self):
        if self.dataloader_val is None:
            return None

        self.model.eval()
        with torch.no_grad():
            loss_sum = torch.zeros(1, device=self.device)
            for i in range(self.val_nr_batch):
                batch = next(self.dataloader_val)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss_sum += loss

        self.model.train()
        return {'val_loss': (loss_sum/self.val_nr_batch).item()}



