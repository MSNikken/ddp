import pypose as pp
import torch
import numpy as np

from typing import Union

from matplotlib import pyplot as plt


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def perturbation(alpha_bar, gamma=1):
    epsilon = torch.randn((len(alpha_bar), 6))
    return pp.se3(gamma * torch.sqrt(1 - alpha_bar[..., None].expand(-1, 6)) * epsilon).Exp()


def interpolate(H1: pp.SE3_type, H2: pp.SE3_type, scale: Union[float, torch.Tensor]):
    return pp.Exp(pp.Log(H2 @ H1.Inv()) * scale[..., None].expand(-1, 6)) @ H1


def interpolate_sqrt_alphas_cumprod(H, sqrt_alphas_cumprod):
    return interpolate(H, pp.identity_like(H, device=H.device), 1 - sqrt_alphas_cumprod)


def diffuse(H0, alpha_bar):
    return perturbation(alpha_bar) @ interpolate_sqrt_alphas_cumprod(H0, torch.sqrt(alpha_bar))


def posterior(H0, Hk, alpha_bar, alpha_bar_prev, beta, beta_tilde, gamma=1):
    lam_0 = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar))[..., None].expand(-1, 6)
    lam_1 = (torch.sqrt(alpha_bar) * (1 - alpha_bar_prev) / (1 - alpha_bar))[..., None].expand(-1, 6)
    noise_scale = (torch.sqrt(beta_tilde) * gamma)[..., None].expand(-1, 6)
    epsilon = torch.randn((len(alpha_bar), 6))
    return (H0.Log() * lam_0 + Hk.Log() * lam_1 + noise_scale * epsilon).Exp()


def diffuse_euc(x0, alpha_bar):
    epsilon = torch.randn((len(alpha_bar), 6))
    return torch.sqrt(alpha_bar)[..., None].expand(-1, 6) * x0 + torch.sqrt(
        1 - alpha_bar[..., None].expand(-1, 6)) * epsilon


def posterior_euc(x0, xk, alpha_bar, alpha_bar_prev, beta, beta_tilde):
    lam_0 = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar))[..., None].expand(-1, 6)
    lam_1 = (torch.sqrt(alpha_bar) * (1 - alpha_bar_prev) / (1 - alpha_bar))[..., None].expand(-1, 6)
    noise_scale = torch.sqrt(beta_tilde)[..., None].expand(-1, 6)
    epsilon = torch.randn((len(alpha_bar), 6))
    return x0 * lam_0 + xk * lam_1 + noise_scale * epsilon


if __name__ == "__main__":
    n_samples = 1000

    H0 = pp.randn_SE3(1)
    H0 = H0.repeat(n_samples, 1)

    x0 = torch.randn((1, 6))
    x0 = x0.repeat(n_samples, 1)

    # Noise parameters
    K = 200

    betas = cosine_beta_schedule(K)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    euc_matches = []
    se3_matches = []
    for k in range(1, K - 1, 1):
        print(f"Starting k={k}")
        beta = betas[k - 1].repeat(n_samples)
        alpha_bar = alphas_cumprod[k - 1].repeat(n_samples)
        alpha_bar_prev = alphas_cumprod_prev[k - 1].repeat(n_samples)
        beta_tilde = ((1 - alpha_bar_prev) * beta / (1 - alpha_bar))
        beta_tilde = torch.zeros_like(beta)

        # Diffusion
        Hk = diffuse(H0, alpha_bar)
        Hk_1 = diffuse(H0, alpha_bar_prev)
        post = posterior(H0, Hk, alpha_bar, alpha_bar_prev, beta, beta_tilde)

        xk = diffuse_euc(x0, alpha_bar)
        xk_1 = diffuse_euc(x0, alpha_bar_prev)
        post_euc = posterior_euc(x0, xk, alpha_bar, alpha_bar_prev, beta, beta_tilde)

        # Comparative analysis
        mean_se3_Hk = torch.mean(Hk.Log(), dim=0)
        mean_se3_Hk_1 = torch.mean(Hk_1.Log(), dim=0)
        mean_se3_post = torch.mean(post.Log(), dim=0)
        se3_match = 1 - ((torch.linalg.vector_norm(mean_se3_Hk_1 - mean_se3_post) /
                          torch.linalg.vector_norm(mean_se3_Hk_1 - mean_se3_Hk)))

        mean_euc_xk = torch.mean(xk, dim=0)
        mean_euc_xk_1 = torch.mean(xk_1, dim=0)
        mean_euc_post = torch.mean(post_euc, dim=0)
        euc_match = 1 - ((torch.linalg.vector_norm(mean_euc_xk_1 - mean_euc_post) /
                          torch.linalg.vector_norm(mean_euc_xk_1 - mean_euc_xk)))

        se3_matches.append(se3_match)
        euc_matches.append(euc_match)

    plt.plot(range(1, K - 1, 1), se3_matches, label='se3')
    plt.plot(range(1, K - 1, 1), euc_matches, label='euc')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('match (1=posterior matches perfectly)')
    plt.show()
    pass
