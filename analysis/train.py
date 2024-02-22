if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    import wandb
    wandb.login()

    sweep = Sweep(RUN, Config).load("default_inv.jsonl")

    for kwargs in sweep:
        with wandb.init(project="diffusion",config=kwargs):
            logger.print(RUN.prefix, color='green')
            jaynes.config("local")
            #thunk = instr(main, **kwargs)
            #jaynes.run(thunk)
            main(**kwargs)

    jaynes.listen()
