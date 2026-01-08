import torch
import torch.nn as nn
from Utils.Net.DDPM_net import UNet

class Model(nn.Module):
    def __init__(self, device, beta_1, beta_T, T):
        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.backbone = UNet(T)

        self.to(device = self.device)

    def loss_fn(self, x, idx=None, T_prime=None):
        output, epsilon, alpha_bar = self.forward(x, idx=idx, get_target=True,T_prime=T_prime)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x,idx=None, get_target=False, T_prime=None):
        if idx == None:
            if T_prime is None:
                T_prime = len(self.alpha_bars)
            idx = torch.randint(0, T_prime, (x.size(0), )).to(device = self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None, None]

            epsilon = torch.randn_like(x)

            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon

        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device = self.device).long()
            x_tilde = x

        output = self.backbone(x_tilde, idx)
        return (output, epsilon, used_alpha_bars) if get_target else output



class DiffusionProcess():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape):

        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape

        self.diffusion_fn = diffusion_fn
        self.device = device


    def _one_diffusion_step(self, x):

        for idx in reversed(range(len(self.alpha_bars))):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            predict_epsilon = self.diffusion_fn(x, idx)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
            x = mu_theta_xt + sqrt_tilde_beta * noise
            yield x

    @torch.no_grad()
    def sampling(self, sampling_number,save_trajectory=True):

        sample = torch.randn([sampling_number,*self.shape]).to(device = self.device)
        sampling_list = []

        final = None
        for sample in self._one_diffusion_step(sample):
            final = sample
            if save_trajectory:
                sampling_list.append(final)

        if save_trajectory:
            trajectory = torch.stack(sampling_list)
            return final, trajectory
        else:
            return final, None