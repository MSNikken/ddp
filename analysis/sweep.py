if __name__ == '__main__':
    import wandb
    import os

    from scripts.train import main
    from config.locomotion_config import Config


    wandb.login()
    mode = os.environ.get("LOGGING", "online")

    sweep_config = {
        'method': 'grid'
    }

    metric = {
        'name': 'loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    # Collect selected config
    parameters_dict = vars(Config)
    parameters_dict = {k: {'value': v} for k, v in parameters_dict.items()}
    # # Select parameters to sweep
    parameters_dict.update({
        # 'kinematic_scale': {
        #     'distribution': 'q_log_uniform_values',
        #     'max': 1001,
        #     'min': 0.1,
        #     'q': 10
        # },
        # 'max_kin_weight': {
        #     'distribution': 'q_log_uniform_values',
        #     'max': 10000,
        #     'min': 1,
        #     'q': 10
        # },
        'kinematic_scale': {
            'values': [500, 10]
        },
        'max_kin_weight': {
            'values': [100000, 10000]
        }
    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['name'] = 'kinloss_pose_sweep'
    sweep_id = wandb.sweep(sweep_config, project="diffusion")

    def train(config=None):
        with wandb.init(project="diffusion", config=config, mode=mode):
            wandb.define_metric("loss", summary="min")
            main()

    wandb.agent(sweep_id, train, count=8)
