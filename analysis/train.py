if __name__ == '__main__':
    import wandb
    import os

    from scripts.train import main
    from config.locomotion_config import Config

    wandb.login()
    mode = os.environ.get("LOGGING", "online")
    with wandb.init(project="diffusion", config=vars(Config), mode=mode):
        wandb.define_metric("loss", summary="min")
        main()
