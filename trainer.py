import torch
from config.train_model_config import NUM_ITER, LR_INIT, get_diffusion_config, get_first_stage_config
from itertools import cycle
from torch.nn.parallel import DistributedDataParallel as DDP
from model.autoencoder import AutoencoderKL
from model.diffusion import GaussianDiffusion
from utils.utils import AverageMeter, compute_grad_param_norms


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, gpu_id):

        self.gpu_id = gpu_id
        self.optimizer = optimizer

        self.vae = AutoencoderKL(**get_first_stage_config()).to(gpu_id)
        self.diffusion = GaussianDiffusion(**get_diffusion_config()).to(gpu_id)
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_anneal_steps = NUM_ITER
        self.step = 0
        self.resume_step = 0
        self.scale_factor = 0.18215
        self.lr = LR_INIT

        self.loss = AverageMeter()
        self.grad_norm = AverageMeter()
        self.param_norm = AverageMeter()

    def start_training(self):
        self._run()

    def _run(self):
        self.train_loader.sampler.set_epoch(self.step)
        data_iterator = cycle(self.train_loader)
        self.model.train()
        while self.step < self.lr_anneal_steps:
            batch = next(data_iterator)
            x = batch['image'].to(self.gpu_id)
            # -----------------
            z = self.encode_image(x)
            z_noisy, t, target = self.get_t_noizyz(z)
            # ----------------
            caption = None # batch['caption']
            self._run_batch(z_noisy, t, target=target, context=caption)
            self.step += 1
            if self.step % len(self.train_loader) == 0:
                self.train_loader.sampler.set_epoch(self.step)
            if self.gpu_id == 0 and self.step % 100 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                print(f'step: {self.step}'
                      f'loss: {self.loss.avg:.4e}'
                      f'grad_norm: {self.grad_norm.avg:.4e}'
                      f'pram: {self.param_norm.avg:.4e}'
                      f'memory_use: {memory_used:.2f}GB')

    def get_t_noizyz(self, z):
        noise = torch.randn_like(z)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.device).long()
        z_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
        return z_noisy, t, noise

    @torch.no_grad()
    def encode_image(self, x):
        z = self.vae.encode(x).sample().detach()
        return self.scale_factor * z

    @torch.no_grad()
    def decode_latent(self, z):
        z = 1. / self.scale_factor * z
        return self.vae.decode(z)

    def _run_batch(self, z_noisy, t, target, context=None):
        self.optimizer.zero_grad()
        pred = self.model(z_noisy, t, context)
        loss = self.compute_loss(pred, target)
        self.backpropagation(loss)


    def backpropagation(self, loss):
        loss.backward()
        self.optimizer.step()
        self.update_metrics(loss.item())
        self.anneal_lr()

    def update_metrics(self, loss):
        grad_norm, param_norm = compute_grad_param_norms(self.model.module)
        self.grad_norm.update(grad_norm)
        self.param_norm.update(param_norm)
        self.loss.update(loss)


    def compute_loss(self, prediction, target, loss_type='l2'):
        if loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(prediction, target)
        elif loss_type == 'l1':
            loss = (target - prediction).abs()
        else:
            raise NotImplementedError
        return loss

    def anneal_lr(self,):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
