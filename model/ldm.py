import torch
import torch.nn as nn
from model.openai_model import UNetModel
from model.autoencoder import AutoencoderKL
from model.diffusion import GaussianDiffusion
from config.train_model_config import get_first_stage_config, get_diffusion_config, get_unet_config, LR_INIT, NUM_ITER



class LDM(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.unet = UNetModel(**get_unet_config())
        self.vae = AutoencoderKL(**get_first_stage_config())
        self.diffusion = GaussianDiffusion(**get_diffusion_config())
        self.conditioner = None
        self.scale_factor = 0.18215

    def forward(self, x, caption=None):
        z = self.get_latent(x)
        noise = torch.randn_like(z)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        z_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
        if self.conditioner is not None:
            pass
        predicted_noise = self.unet(z_noisy, t, context=None)
        return predicted_noise, noise

    @torch.no_grad()
    def get_latent(self, x):
        z = self.vae.encode(x).sample().detach()
        return z * self.scale_factor
    @torch.no_grad()
    def get_image_from_latent(self, z):
        z = 1. / self.scale_factor * z
        return self.vae.decode(z)
