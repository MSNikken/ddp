import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from diffuser import utils
from diffuser.datasets import Zone
from diffuser.datasets.reward import discounted_trajectory_rewards
from diffuser.experiments.samples import inpaint_scenarios
from diffuser.utils import plot_trajectory, plot_trajectory_2d, draw_rectangles, plot_trajectory_summary_2d

#runpath = 'dl_rob/diffusion/l9z7ryr8' sleek donkey
#runpath = 'dl_rob/diffusion/vvnf5fm3'  # rich sweep 1
#runpath = 'dl_rob/diffusion/fkfi36xg'  # comic totem 196 (hor32, 0,randcond)
#runpath = 'dl_rob/diffusion/2dr0kadq'  # sleek surf (comic + kin)
#runpath = 'dl_rob/diffusion/h0f1ywab'  # skilled violet (comic + dense)
#runpath = 'dl_rob/diffusion/gcf9cf3d'  # bright dawn (comic + dense + kin)

#runpath = 'dl_rob/diffusion/lz32b97k' # electric field (kdbr)

runpaths = {
    'ddb': 'dl_rob/diffusion/i2ff19p3',
    'kdb': 'dl_rob/diffusion/m8isvw5o',
    'ddbr': 'dl_rob/diffusion/u3i0kjj8',
    'kdbr': 'dl_rob/diffusion/vei4ird6',
    'ddbf': 'dl_rob/diffusion/8w3vif8e',
    'kdbf': 'dl_rob/diffusion/a4leduhr',
    'ddbrf': 'dl_rob/diffusion/x1xp7na1',
    'kdbrf': 'dl_rob/diffusion/h9fk00mt',
    'ddb2': 'dl_rob/diffusion/5t0sxiys',
    'kdb2': 'dl_rob/diffusion/xpm2hkj3',
    'ddbr2': 'dl_rob/diffusion/1f6z3xeb',
    'kdbr2': 'dl_rob/diffusion/stnlg8gn',
    'ddbf2': 'dl_rob/diffusion/j4xxf9oy',
    'kdbf2': 'dl_rob/diffusion/eqvxqf7s',
    'ddbrf2': 'dl_rob/diffusion/39nhtqjg',
    'kdbrf2': 'dl_rob/diffusion/j7bsphez',
    'dsb2': 'dl_rob/diffusion/mynpt464',
    'dsb3': 'dl_rob/diffusion/eddiiugp',
    'ddb3': 'dl_rob/diffusion/32rdz3m8',

    'scn1_dense': 'dl_rob/diffusion/eiojupzx',
    'scn1_sparse': 'dl_rob/diffusion/bhi4i7vh',
    'scn2_dense': 'dl_rob/diffusion/5wqqhflh',
    'scn2_sparse': 'dl_rob/diffusion/fv6dvdp2',
    'scn3_dense': 'dl_rob/diffusion/s9qxlvqh',
    'scn3_sparse': 'dl_rob/diffusion/rczqxeiz',
}
runpath = runpaths['scn3_sparse']


# Retrieve configuration
api = wandb.Api()

run = api.run(runpath)
wb_config = run.config

# Retrieve parameters
state_file = wandb.restore('checkpoint/state.pt', run_path=runpath)
state_dict = torch.load(state_file.name)
ema_dict = state_dict['ema']


if __name__ == "__main__":
    from config.locomotion_config import Config
    Config._update(wb_config)
    #Config._update({'condition_guidance_w': 1.2})   # 1.2 default
    pt_file = state_file.name
    print(f'Retrieving parameters from: {pt_file}')
    experiment = utils.serialization.load_diffusion_from_config(Config, pt_file)
    diff_model = experiment.ema
    dataset = experiment.dataset['train']

    device = 'cuda'
    # Scenarios to plot:

    # Default obstacle
    # scns = ['bme_straight', 'bme_straight_long']
    # obstacles = [[0.4, 0.4, 0.6, 0.6]]
    # view = 'xy'

    # Franka obstacle
    #scns = ['fr_be_straight', 'fr_be_short', 'fr_be_short2', 'fr_bm_straight', 'fr_be_straight2', 'fr_be_straight3']
    #scns = ['fr_be_short', 'fr_be_short2', 'fr_be_short3', 'fr_be_short4', 'fr_be_short5']
    #scns = ['fr_be_straight', 'fr_be_straight2', 'fr_be_straight3', 'fr_be_straight4']
    #scns = ['fr_bm_straight', 'fr_bm_straight2']
    #scns = ['fr_be_short', 'fr_be_longer']
    #scns = ['fr_be_parallel']
    #scns = ['fr_b']
    #scns = ['fr_be_straight', 'fr_be_straight4']
    #scns = ['fr_curve']

    xlim = (0.3, 0.5)
    ylim = (-0.25, 0.25)
    zlim = (0.3, 0.6)

    #scns = ['scn2_1', 'scn2_2', 'scn2_3', 'scn2_4']
    scns = ['scn3_1', 'scn3_2']

    # Scenarios:
    obstacles1 = [Zone(xmin=0.3, ymin=-0.1, zmin=0.3, xmax=0.5, ymax=0.1, zmax=0.45)]
    views1 = ['yz']
    lims1 = [(ylim, zlim)]
    scn_rectangles1 = [[[obst.ymin, obst.zmin, obst.ymax, obst.zmax] for obst in obstacles1]]

    obstacles2 = [
        Zone(xmin=0.3, ymin=-0.15, zmin=0.3, xmax=0.41, ymax=-0.05, zmax=0.6),
        Zone(xmin=0.39, ymin=0.05, zmin=0.3, xmax=0.5, ymax=0.15, zmax=0.6)]
    views2 = ['xy']
    lims2 = [(xlim, ylim)]
    scn_rectangles2 = [[[obst.xmin, obst.ymin, obst.xmax, obst.ymax] for obst in obstacles2]]

    obstacles3 = [
        Zone(xmin=0.3, ymin=0.0, zmin=0.3, xmax=0.5, ymax=0.25, zmax=0.4),
        Zone(xmin=0.3, ymin=-0.05, zmin=0.3, xmax=0.4, ymax=0.05, zmax=0.6)]
    views3 = ['xy', 'yz']
    lims3 = [(xlim, ylim), (ylim, zlim)]
    scn_rectangles3 = [[[obst.xmin, obst.ymin, obst.xmax, obst.ymax] for obst in obstacles3],
                       [[obst.ymin, obst.zmin, obst.ymax, obst.zmax] for obst in obstacles3]]

    obstacles = obstacles3
    scn_rectangles = scn_rectangles3
    views = views3
    lims = lims3


    # plotting
    step = 1
    remove_duplicates = True
    grid = False

    # diffusion
    n_samples = 30
    horizon = [128]
    inference_returns = [0]

    if grid:
        figs = [plt.figure() for i in range(len(scns))]
    for i_hor, hor in enumerate(horizon):
        for i_ret, ret in enumerate(inference_returns):
            # List of tuple of (final, whole diffusion)
            diff = inpaint_scenarios(diff_model, dataset, hor, scns, device, n_samples=n_samples,
                                     inference_returns=ret, return_diff=True, unnorm=True)
            for i_scn, (scn, scn_name) in enumerate(zip(diff, scns)):
                # Metrics
                rew = discounted_trajectory_rewards(torch.tensor(scn[0]), obstacles, discount=0.99, kin_rel_weight=0)
                collision = discounted_trajectory_rewards(torch.tensor(scn[0]), obstacles, discount=1, kin_rel_weight=0)
                kin_score = discounted_trajectory_rewards(torch.tensor(scn[0]), [], discount=1, kin_rel_weight=1, kin_l1=True)
                collision_free = (collision == 0).sum()
                kin_score_mean = kin_score.mean()
                dist_to_end = np.linalg.norm(scn[0][:, -1, :3] - scn[0][:, -2, :3], axis=-1)
                dist_to_end_mean = np.mean(dist_to_end)

                print(f"Metrics {scn_name}:"
                      f"\n Reward mean: {rew.mean()}"
                      f"\n Collision free trajectories: {collision_free}/{n_samples}"
                      f"\n Mean kinematic score: {kin_score_mean}"
                      f"\n Mean distance to end: {dist_to_end_mean}")

                # Plotting
                for i in [199]:
                    traj = scn[1][0, i]
                    #plot_trajectory(traj, block=False, as_equal=True, step=step)
                    #fig, ax = plot_trajectory_2d(traj, view=view, block=False, as_equal=True, step=step)
                    # draw_rectangles(ax, rectangles)
                    # ax.set_xlim(*xlim)
                    # ax.set_ylim(*ylim)

                paths = scn[0]
                if remove_duplicates:
                    non_duplicate = np.concatenate((np.array([True]), (paths[0, 1:] != paths[0, :-1]).all(-1)))
                    paths = np.ascontiguousarray(paths[:, non_duplicate])

                for view, rectangles, lim in zip(views, scn_rectangles, lims):
                    if grid:
                        fig = figs[i_scn]
                        ax = fig.add_subplot(len(horizon), len(inference_returns), i_hor*len(inference_returns) + i_ret+1)
                        ax = plot_trajectory_summary_2d(paths, view=view, block=False, as_equal=True, labels=False, ax=ax)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                        if i_hor == len(horizon)-1:
                            ax.set_xlabel('y (m)')
                        if i_ret == 0:
                            ax.set_ylabel('z (m)')
                    else:
                        fig = plt.figure()
                        ax = fig.subplots()
                        ax = plot_trajectory_summary_2d(paths, view=view, block=False, as_equal=True, labels=True, ax=ax)
                        #ax.set_title('Sampling with sparse rewards')
                        ax.legend()
                        ax.set_xlabel('y (m)')
                        ax.set_ylabel('z (m)')

                    draw_rectangles(ax, rectangles)
                    ax.set_xlim(*lim[0])
                    ax.set_ylim(*lim[1])
                    # ax.set_title(scn_name)

plt.show()
