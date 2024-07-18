import os

import torch
import yaml

import diff_planner.diffuser.utils as utils

from params_proto.neo_proto import ParamsProto


class Config(ParamsProto):
    # misc
    seed = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bucket = os.path.normpath(os.path.join(os.getcwd(), os.pardir, 'weights'))
    dataset = 'hopper-medium-expert-v2'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'
    horizon = 100
    n_diffusion_steps = 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9

    ## rendering
    renderer = 'utils.MuJoCoRenderer'
    representation = 'joint'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 400.0   # Determined using rewards from the dataset
    dt = 0.08   # time step in seconds
    pose_only = False

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'state_l2'
    n_train_steps = 1e6
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False
    train_kinematic_loss = False
    kinematic_loss_type = None
    kinematic_scale = 0
    max_kin_weight = 1e5
    kin_weight_cutoff = -1
    kin_norm = False
    train_data_loss = True


def load_diff_model(config_file, pt_file):
    if config_file is not None:
        with open(config_file) as file:
            override = yaml.safe_load(file)
            for k, v in override.items():
                setattr(Config, k, v)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        repres=Config.representation
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
        repres=Config.representation
    )

    dataset = dataset_config()
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_diffsteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            hidden_dim=Config.hidden_dim,
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            # Kinematic loss
            train_kinematic_loss=Config.train_kinematic_loss,
            kinematic_loss_type=Config.kinematic_loss_type,
            kinematic_scale=Config.kinematic_scale,
            max_kin_weight=Config.max_kin_weight,
            kin_weight_cutoff=Config.kin_weight_cutoff,
            dt=Config.dt,
            pose_only=Config.pose_only,
            train_data_loss=Config.train_data_loss,
            device=torch.device(Config.device),
        )
    elif Config.diffusion == 'models.SE3Diffusion':
        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_diffsteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            hidden_dim=Config.hidden_dim,
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            # noise scaling
            gamma=Config.gamma,
            # loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            # Kinematic loss
            kinematic_loss=Config.kinematic_loss,
            kinematic_scale=Config.kinematic_scale,
            max_kin_weight=Config.max_kin_weight,
            dt=Config.dt,
            device=torch.device(Config.device),
        )
    else:
        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_diffsteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            # Kinematic loss
            kinematic_loss=Config.kinematic_loss,
            kinematic_scale=Config.kinematic_scale,
            max_kin_weight=Config.max_kin_weight,
            dt=Config.dt,
            device=torch.device(Config.device),
        )

    if Config.diffusion == 'models.GaussianInvDynDiffusion' or Config.diffusion == 'models.SE3Diffusion':
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=torch.device(Config.device),
        )
    else:
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=torch.device(Config.device),
        )

    # trainer_config = utils.Config(
    #     utils.Trainer,
    #     savepath='trainer_config.pkl',
    #     train_batch_size=Config.batch_size,
    #     train_lr=Config.learning_rate,
    #     gradient_accumulate_every=Config.gradient_accumulate_every,
    #     ema_decay=Config.ema_decay,
    #     sample_freq=Config.sample_freq,
    #     save_freq=Config.save_freq,
    #     log_freq=Config.log_freq,
    #     label_freq=int(Config.n_train_steps // Config.n_saves),
    #     save_parallel=Config.save_parallel,
    #     bucket=Config.bucket,
    #     n_reference=Config.n_reference,
    #     train_device=torch.device(Config.device),
    #     save_checkpoints=Config.save_checkpoints,
    # )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model, normalizer=dataset.normalizer)
    diffusion.load_state_dict(torch.load(pt_file)['ema'])

    # trainer = trainer_config(diffusion, dataset, renderer)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    print('Testing forward/backward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print('✓')
    return diffusion
