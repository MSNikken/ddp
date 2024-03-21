if __name__ == '__main__':
    from scripts.train import main
    from config.locomotion_config import Config
    import wandb

    wandb.login()

    with wandb.init(project="diffusion", config=vars(Config), mode="disabled"):
        wandb.define_metric("loss", summary="min")
        main()
