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
        'dataset': {
            'values': [
                'datasets.FrankaLinesPoseObstDense3',
                'datasets.FrankaLinesPoseObstSparse3'
            ]
        },

    })
    sweep_config['parameters'] = parameters_dict
    sweep_config['name'] = 'conditional_sampling'
    sweep_id = wandb.sweep(sweep_config, project="diffusion")

    def train(config=None):
        with wandb.init(project="diffusion", config=config, mode=mode):
            wandb.define_metric("loss", summary="min")
            main(**vars(wandb.config)['_items'])

    wandb.agent(sweep_id, train, count=8)
