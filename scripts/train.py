import diffuser.utils as utils
import torch


def main(**deps):
    from config.locomotion_config import Config

    Config._update(deps)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.load_dataset_config(Config)
    dataset_val_config = utils.load_dataset_config(Config, val=True)
    render_config = utils.load_renderer_config(Config)

    dataset = dataset_config()
    dataset_val = dataset_val_config() if Config.dataset_val is not None else None
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    returns_dim = dataset.returns_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    diffusion_config = utils.load_diffuser_config(Config, observation_dim, action_dim)

    model_config = utils.load_model_config(Config, observation_dim, action_dim, returns_dim)

    trainer_config = utils.load_trainer_config(Config)

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model, normalizer=dataset.normalizer)

    trainer = trainer_config(diffusion, dataset, renderer, dataset_val=dataset_val)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    print('Testing forward/backward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print('✓')

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch)

