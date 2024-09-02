import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

from .config import Config
from .training import Trainer

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')


def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config


def load_diffusion(*loadpath, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)


def load_dataset_config(config, val=False):
    dataset_config = Config(
        config.loader,
        savepath='dataset_config.pkl' if not val else 'dataset_config_val.pkl',
        env=config.dataset if not val else config.dataset_val,
        horizon=config.horizon,
        normalizer=config.normalizer,
        preprocess_fns=config.preprocess_fns,
        use_padding=config.use_padding,
        max_path_length=config.max_path_length,
        max_n_episodes=config.max_n_episodes,
        condition_indices=config.condition_indices,
        include_returns=config.include_returns,
        returns_scale=config.returns_scale,
        discount=config.discount,
        termination_penalty=config.termination_penalty,
        repres=config.representation
    )
    return dataset_config


def load_renderer_config(config):
    render_config = Config(
        config.renderer,
        savepath='render_config.pkl',
        env=config.dataset,
        repres=config.representation
    )
    return render_config


def load_model_config(config, obs_dim, act_dim, ret_dim):
    if config.diffusion == 'models.GaussianInvDynDiffusion' or config.diffusion == 'models.SE3Diffusion':
        model_config = Config(
            config.model,
            savepath='model_config.pkl',
            horizon=config.horizon,
            transition_dim=obs_dim,
            returns_dim=ret_dim,
            cond_dim=obs_dim,
            dim_mults=config.dim_mults,
            returns_condition=config.returns_condition,
            dim=config.dim,
            condition_dropout=config.condition_dropout,
            calc_energy=config.calc_energy,
            device=torch.device(config.device),
        )
    else:
        model_config = Config(
            config.model,
            savepath='model_config.pkl',
            horizon=config.horizon,
            transition_dim=obs_dim + act_dim,
            returns_dim=ret_dim,
            cond_dim=obs_dim,
            dim_mults=config.dim_mults,
            returns_condition=config.returns_condition,
            dim=config.dim,
            condition_dropout=config.condition_dropout,
            calc_energy=config.calc_energy,
            device=torch.device(config.device),
        )
    return model_config


def load_diffuser_config(config, obs_dim, act_dim):
    if config.diffusion == 'models.GaussianInvDynDiffusion':
        diffusion_config = Config(
            config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=config.horizon,
            observation_dim=obs_dim,
            action_dim=act_dim,
            n_diffsteps=config.n_diffusion_steps,
            loss_type=config.loss_type,
            clip_denoised=config.clip_denoised,
            predict_epsilon=config.predict_epsilon,
            hidden_dim=config.hidden_dim,
            ar_inv=config.ar_inv,
            train_only_inv=config.train_only_inv,
            ## loss weighting
            action_weight=config.action_weight,
            loss_weights=config.loss_weights,
            loss_discount=config.loss_discount,
            returns_condition=config.returns_condition,
            condition_guidance_w=config.condition_guidance_w,
            # Kinematic loss
            train_kinematic_loss=config.train_kinematic_loss,
            kinematic_loss_type=config.kinematic_loss_type,
            kinematic_scale=config.kinematic_scale,
            max_kin_weight=config.max_kin_weight,
            kin_weight_cutoff=config.kin_weight_cutoff,
            dt=config.dt,
            pose_only=config.pose_only,
            train_data_loss=config.train_data_loss,
            device=torch.device(config.device),
        )
    elif config.diffusion == 'models.SE3Diffusion':
        diffusion_config = Config(
            config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=config.horizon,
            observation_dim=obs_dim,
            action_dim=act_dim,
            n_diffsteps=config.n_diffusion_steps,
            loss_type=config.loss_type,
            clip_denoised=config.clip_denoised,
            predict_epsilon=config.predict_epsilon,
            hidden_dim=config.hidden_dim,
            ar_inv=config.ar_inv,
            train_only_inv=config.train_only_inv,
            # noise scaling
            gamma=config.gamma,
            # loss weighting
            action_weight=config.action_weight,
            loss_weights=config.loss_weights,
            loss_discount=config.loss_discount,
            returns_condition=config.returns_condition,
            condition_guidance_w=config.condition_guidance_w,
            # Kinematic loss
            kinematic_loss=config.kinematic_loss,
            kinematic_scale=config.kinematic_scale,
            max_kin_weight=config.max_kin_weight,
            dt=config.dt,
            device=torch.device(config.device),
        )
    else:
        diffusion_config = Config(
            config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=config.horizon,
            observation_dim=obs_dim,
            action_dim=act_dim,
            n_diffsteps=config.n_diffusion_steps,
            loss_type=config.loss_type,
            clip_denoised=config.clip_denoised,
            predict_epsilon=config.predict_epsilon,
            ## loss weighting
            action_weight=config.action_weight,
            loss_weights=config.loss_weights,
            loss_discount=config.loss_discount,
            returns_condition=config.returns_condition,
            condition_guidance_w=config.condition_guidance_w,
            # Kinematic loss
            kinematic_loss=config.kinematic_loss,
            kinematic_scale=config.kinematic_scale,
            max_kin_weight=config.max_kin_weight,
            dt=config.dt,
            device=torch.device(config.device),
        )
    return diffusion_config


def load_trainer_config(config):
    trainer_config = Config(
        Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=config.batch_size,
        train_lr=config.learning_rate,
        gradient_accumulate_every=config.gradient_accumulate_every,
        ema_decay=config.ema_decay,
        sample_freq=config.sample_freq,
        save_freq=config.save_freq,
        log_freq=config.log_freq,
        label_freq=int(config.n_train_steps // config.n_saves),
        save_parallel=config.save_parallel,
        bucket=config.bucket,
        n_reference=config.n_reference,
        inference_returns=config.inference_returns,
        inference_horizon=config.inference_horizon,
        train_device=torch.device(config.device),
        save_checkpoints=config.save_checkpoints,
        val_batch_size=config.val_batch_size,
        val_nr_batch=config.val_nr_batch,
    )
    return trainer_config


def load_diffusion_from_config(config, loadpath):
    dataset = {'train': load_dataset_config(config)(), 'val': None}
    if hasattr(config, 'dataset_val') and config.dataset_val is not None:
        dataset['val'] = load_dataset_config(config, val=True)()
    normalizer = dataset['train'].normalizer
    obs_dim = dataset['train'].observation_dim
    act_dim = dataset['train'].action_dim
    ret_dim = dataset['train'].returns_dim
    renderer = load_renderer_config(config)()
    model = load_model_config(config, obs_dim, act_dim, ret_dim)()
    diffusion = load_diffuser_config(config, obs_dim, act_dim)(model, normalizer=normalizer)
    trainer = load_trainer_config(config)(diffusion, dataset, renderer, dataset['val'])

    state = torch.load(loadpath)
    diffusion.load_state_dict(state['model'])
    trainer.ema_model.load_state_dict(state['ema'])
    trainer.step = state['step']
    epoch = -1
    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)
