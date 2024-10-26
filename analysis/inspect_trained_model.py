import os
import pickle
from functools import partial

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from diffuser import utils
from diffuser.datasets import Zone
from diffuser.datasets.reward import discounted_trajectory_rewards, cost_collision, cost_ee
from diffuser.experiments.samples import inpaint_scenarios
from diffuser.models.guide import CostComposite, GuideManagerTrajectories
from diffuser.utils import plot_trajectory, plot_trajectory_2d, draw_rectangles, plot_trajectory_summary_2d


runpaths = {
    'example_path': 'wandb_group/project/run',

}
runpath = runpaths['example_path']

# Retrieve configuration
api = wandb.Api()

run = api.run(runpath)
wb_config = run.config

# Retrieve parameters
print('Loading parameters')
state_file = wandb.restore('checkpoint/state.pt', run_path=runpath)

if __name__ == "__main__":
    from config.locomotion_config import Config

    Config._update(wb_config)
    pt_file = state_file.name
    print(f'Retrieving parameters from: {pt_file}')
    experiment = utils.serialization.load_diffusion_from_config(Config, pt_file)
    diff_model = experiment.ema
    dataset = experiment.dataset['train']
    device = 'cuda'
    # Scenarios to plot:


    # Plot limits
    xlim = (0.3, 0.5)
    ylim = (-0.25, 0.25)
    zlim = (0.3, 0.6)

    # Franka obstacle
    scns = ['scn1_1', 'scn1_repeat2', 'scn1_repeat5', 'scn1_repeat10']
    # Scenarios:
    obstacles = [Zone(xmin=0.3, ymin=-0.1, zmin=0.3, xmax=0.5, ymax=0.1, zmax=0.45)]
    views = ['yz']
    lims = [(ylim, zlim)]
    scn_rectangles = [[[obst.ymin, obst.zmin, obst.ymax, obst.zmax] for obst in obstacles]]

    # Cost guided sampling
    # end effector costs are added when goal pose is known
    costs = [CostComposite([partial(cost_collision, obstacles), cost_ee], weights=[1e-1, 1e-2]) for scn in scns]
    guides = [GuideManagerTrajectories(cost, dataset.normalizer, clip_grad=True) for cost in costs]
    #guies = None

    # plotting
    step = 1
    remove_duplicates = True
    grid = False
    save_pickle = False

    # diffusion
    n_samples = 30
    horizon = [128]
    inference_returns = [-0.001]

    if grid:
        figs = [plt.figure() for i in range(len([0]))]
    for i_hor, hor in enumerate(horizon):
        for i_ret, ret in enumerate(inference_returns):
            # List of tuple of (final, whole diffusion)
            diff = inpaint_scenarios(diff_model, dataset, hor, scns, device, n_samples=n_samples,
                                     inference_returns=ret, return_diff=True, unnorm=True, guides=guides)
            for i_scn, (scn, scn_name) in enumerate(zip(diff, scns)):
                # Some relevant metrics
                rew = discounted_trajectory_rewards(torch.tensor(scn[0]), obstacles, discount=0.99, kin_rel_weight=0)
                collision = discounted_trajectory_rewards(torch.tensor(scn[0]), obstacles, discount=1, kin_rel_weight=0)
                kin_score = discounted_trajectory_rewards(torch.tensor(scn[0]), [], discount=1, kin_rel_weight=1,
                                                          kin_l1=True)
                collision_free = (collision == 0).sum()
                kin_score_mean = kin_score.mean()
                dist_to_end = np.linalg.norm(scn[0][:, -1, :3] - scn[0][:, -2, :3], axis=-1)
                dist_to_end_mean = np.mean(dist_to_end)

                print(f"Metrics {scn_name}:"
                      f"\n Reward mean: {rew.mean()}"
                      f"\n Collision free trajectories: {collision_free}/{n_samples}"
                      f"\n Mean kinematic score: {kin_score_mean}"
                      f"\n Mean distance to end: {dist_to_end_mean}")

                # Plotting various diffusion levels
                for i in [199]:
                    traj = scn[1][0, i]
                    # plot_trajectory(traj, block=False, as_equal=True, step=step)
                    # fig, ax = plot_trajectory_2d(traj, view=view, block=False, as_equal=True, step=step)
                    # draw_rectangles(ax, rectangles)
                    # ax.set_xlim(*xlim)
                    # ax.set_ylim(*ylim)

                paths = scn[0]
                if remove_duplicates:
                    non_duplicate = np.concatenate((np.array([True]), (paths[0, 1:] != paths[0, :-1]).all(-1)))
                    paths = np.ascontiguousarray(paths[:, non_duplicate])

                for view, rectangles, lim in zip(views, scn_rectangles, lims):
                    if grid:

                        # # (h,r)
                        # n_rows = len(horizon)
                        # n_cols = len(inference_returns)
                        # i_row = i_hor
                        # i_col = i_ret
                        # i_fig = i_scn
                        # fig = figs[i_fig]

                        # (h,s)
                        n_rows = len(horizon)
                        n_cols = len(scns)
                        i_row = i_hor
                        i_col = i_scn
                        fig = figs[0]

                        ax = fig.add_subplot(n_rows, n_cols, i_row * n_cols + i_col + 1)
                        ax = plot_trajectory_summary_2d(paths, view=view, block=False, as_equal=True, labels=False,
                                                        ax=ax)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                        if i_row == n_rows - 1:
                            ax.set_xlabel('y (m)')
                        if i_col == 0:
                            ax.set_ylabel('z (m)')
                    else:
                        fig = plt.figure()
                        ax = fig.subplots()
                        ax = plot_trajectory_summary_2d(paths, view=view, block=False, as_equal=True, labels=True,
                                                        ax=ax)
                        ax.legend()
                        ax.set_xlabel('y (m)')
                        ax.set_ylabel('z (m)')

                    draw_rectangles(ax, rectangles)
                    ax.set_xlim(*lim[0])
                    ax.set_ylim(*lim[1])

                    if save_pickle:
                        pickle.dump(fig, open(f'fig{i_scn}_{i_hor}_{i_ret}.pkl', 'wb'))

plt.show()
