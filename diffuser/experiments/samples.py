from functools import partial

import einops
import torch

from diffuser.datasets.reward import cost_ee
from diffuser.utils import to_device, apply_dict, to_np

inpaint_scns = {
    'b': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []}
    },
    'be_straight': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        -1: {'pos': [0.3, 0.35, 0.5], 'rot': []}
    },
    'bm_straight': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.4, 0.35, 0.5], 'rot': []}
    },
    'bme_straight': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.45, 0.35, 0.5], 'rot': []},
        -1: {'pos': [0.4, 0.35, 0.5], 'rot': []}
    },
    'bme_straight_long': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.4, 0.35, 0.5], 'rot': []},
        -1: {'pos': [0.3, 0.35, 0.5], 'rot': []}
    },
    'be_corner': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        -1: {'pos': [0.3, 0.45, 0.5], 'rot': []}
    },
    'bm_corner': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.5, 0.35, 0.5], 'rot': []}
    },
    'bme_corner': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.3, 0.35, 0.5], 'rot': []},
        -1: {'pos': [0.3, 0.45, 0.5], 'rot': []}
    },
    'be_conflict': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        -1: {'pos': [0.5, 0.45, 0.5], 'rot': []}
    },
    'bm_conflict': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.5, 0.4, 0.5], 'rot': []},
    },
    'bme_conflict': {
        0: {'pos': [0.5, 0.35, 0.5], 'rot': []},
        'mid': {'pos': [0.5, 0.4, 0.5], 'rot': []},
        -1: {'pos': [0.5, 0.45, 0.5], 'rot': []}
    },
    'fr_be_straight': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, 0.15, 0.4], 'rot': []}
    },
    'fr_be_short': {
            0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
            -1: {'pos': [0.4, -0.15, 0.5], 'rot': []}
        },
    'fr_be_short2': {
            0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
            -1: {'pos': [0.4, -0.15, 0.5], 'rot': []},
            -2: {'pos': [0.4, -0.15, 0.5], 'rot': []}
        },
    'fr_be_short3': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -2: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -3: {'pos': [0.4, -0.15, 0.5], 'rot': []}
    },
    'fr_be_short4': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -2: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -3: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -4: {'pos': [0.4, -0.15, 0.5], 'rot': []}
    },
    'fr_be_short5': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -2: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -3: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -4: {'pos': [0.4, -0.15, 0.5], 'rot': []},
        -5: {'pos': [0.4, -0.15, 0.5], 'rot': []}
    },
    'fr_be_longer': {
            0: {'pos': [0.4, -0.15, 0.3], 'rot': []},
            -1: {'pos': [0.4, -0.15, 0.5], 'rot': []}
        },
    'fr_bm_straight': {
            0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
            'mid': {'pos': [0.4, -0.15, 0.5], 'rot': []}
        },
    'fr_bm_straight2': {
            0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
            28: {'pos': [0.4, -0.15, 0.5], 'rot': []},
            29: {'pos': [0.4, -0.15, 0.5], 'rot': []},
            30: {'pos': [0.4, -0.15, 0.5], 'rot': []},
            31: {'pos': [0.4, -0.15, 0.5], 'rot': []},
            32: {'pos': [0.4, -0.15, 0.5], 'rot': []}
        },
    'fr_be_straight2': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -2: {'pos': [0.4, 0.15, 0.4], 'rot': []}
    },
    'fr_be_straight3': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -2: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -3: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -4: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -5: {'pos': [0.4, 0.15, 0.4], 'rot': []}
    },
    'fr_be_straight4': {
        0: {'pos': [0.4, -0.15, 0.4], 'rot': []},
        -1: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -2: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -3: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -4: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -5: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -6: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -7: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -8: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -9: {'pos': [0.4, 0.15, 0.4], 'rot': []},
        -10: {'pos': [0.4, 0.15, 0.4], 'rot': []}
    },
    'fr_be_parallel': {
        0: {'pos': [0.4, -0.1, 0.5], 'rot': []},
        -1: {'pos': [0.4, 0.1, 0.5], 'rot': []}
    },
    'fr_b': {
        0: {'pos': [0.4, -0.1, 0.5], 'rot': []}
    },
    'fr_curve': {
        0: {'pos': [0.4, -0.2, 0.35], 'rot': []},
        -1: {'pos': [0.4, 0.2, 0.35], 'rot': []}
    },
    'scn1_1': {
        0: {'pos': [0.4, -0.2, 0.35], 'rot': []},
        -1: {'pos': [0.4, 0.2, 0.35], 'rot': []}
    },
    'scn2_1': {
        0: {'pos': [0.4, -0.2, 0.45], 'rot': []},
        -1: {'pos': [0.4, 0.2, 0.45], 'rot': []}
    },
    'scn2_2': {
        0: {'pos': [0.4, -0.2, 0.45], 'rot': []},
        -1: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -2: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -3: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -4: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -5: {'pos': [0.4, 0.2, 0.45], 'rot': []},
    },
    'scn2_3': {
        0: {'pos': [0.4, 0.0, 0.45], 'rot': []},
        -1: {'pos': [0.4, 0.2, 0.45], 'rot': []}
    },
    'scn2_4': {
        0: {'pos': [0.4, 0.0, 0.45], 'rot': []},
        -1: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -2: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -3: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -4: {'pos': [0.4, 0.2, 0.45], 'rot': []},
        -5: {'pos': [0.4, 0.2, 0.45], 'rot': []},
    },
    'scn3_1': {
        0: {'pos': [0.35, -0.2, 0.35], 'rot': []},
        -1: {'pos': [0.35, 0.2, 0.45], 'rot': []}
    },
    'scn3_2': {
        0: {'pos': [0.35, -0.2, 0.35], 'rot': []},
        -1: {'pos': [0.35, 0.2, 0.45], 'rot': []},
        -2: {'pos': [0.35, 0.2, 0.45], 'rot': []},
        -3: {'pos': [0.35, 0.2, 0.45], 'rot': []},
        -4: {'pos': [0.35, 0.2, 0.45], 'rot': []},
        -5: {'pos': [0.35, 0.2, 0.45], 'rot': []}
    }

}


def get_conditions(scn, normalizer, horizon, obs_dim, device='cpu'):
    conditions = {}
    for k, v in inpaint_scns[scn].items():
        condition = torch.rand(1, obs_dim, device=device) * 2 - 1
        condition = normalizer.unnormalize(condition, 'observations')
        if len(v['pos']) == 3:
            condition[:, :3] = torch.tensor(v['pos'], device=device)
        if len(v['rot']) == 4:
            condition[:, 3:7] = torch.tensor(v['rot'], device=device)
        condition = normalizer.normalize(condition, 'observations')
        i = int(horizon / 2) if k == 'mid' else k
        conditions[i] = condition
    return conditions


def get_samples(model, dataset, conditions, returns, horizon, device, n_samples=2, unnorm=False, return_diff=False,
                guide=None):
    if model.returns_condition and returns is not None:
        returns = to_device(torch.ones(n_samples, dataset.returns_dim, device=device) * returns, device)
    else:
        returns = None

    if model.model.calc_energy:
        samples = model.grad_conditional_sample(conditions, returns=returns, horizon=horizon,
                                                return_diffusion=return_diff)
    else:
        samples = model.conditional_sample(conditions, returns=returns, horizon=horizon,
                                           return_diffusion=return_diff, guide=guide)

    if not return_diff:
        samples = to_np(samples)
        if unnorm:
            samples = dataset.normalizer.unnormalize(samples, 'observations')
        return samples

    samples_final = to_np(samples[0])
    samples_diff = to_np(samples[1])
    if unnorm:
        samples_final = dataset.normalizer.unnormalize(samples_final, 'observations')
        samples_diff = dataset.normalizer.unnormalize(samples_diff, 'observations')
    return samples_final, samples_diff


def inpaint_scenarios(model, dataset, horizon, scns, device, inference_returns=None, n_samples=2, unnorm=True,
                      return_diff=False, guides=None):
    returns = None if inference_returns is None else (
        to_device(torch.ones(n_samples, dataset.returns_dim)*inference_returns, device))
    guides = guides if guides is not None else [None] * len(scns)
    results = []
    for scn, guide in zip(scns, guides):
        conditions = get_conditions(scn, dataset.normalizer, horizon, 7, device=device)

        # Update guide to match goal
        if guide is not None:
            for i in range(len(guide.cost.cost_l)):
                if guide.cost.cost_l[i] == cost_ee:
                    guide.cost.cost_l[i] = partial(cost_ee, dataset.normalizer.unnormalize(conditions[-1], 'observations'))

        # repeat each item in conditions `n_samples` times
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'b d -> (repeat b) d', repeat=n_samples,
        )

        res = get_samples(model, dataset, conditions, returns, horizon, device, n_samples=n_samples,
                          unnorm=unnorm, return_diff=return_diff, guide=guide)
        results.append(res)
    return results
