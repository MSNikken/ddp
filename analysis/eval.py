if __name__ == '__main__':
    from scripts.evaluate_inv_parallel import evaluate
    from config.locomotion_config import Config
    import wandb

    wandb.login()

    with wandb.init(project="diffusion_eval", config=vars(Config)):
        wandb.define_metric("loss", summary="min")
        evaluate()
