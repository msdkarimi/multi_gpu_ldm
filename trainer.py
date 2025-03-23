import torch
from config.train_model_config import NUM_ITER, LR_INIT
from itertools import cycle
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, gpu_id):

        self.gpu_id = gpu_id
        self.optimizer = optimizer
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_anneal_steps = NUM_ITER
        self.step = 0
        self.resume_step = 0
        self.lr = LR_INIT

    def start_training(self):
        self._run()

    def _run(self):
        data_iterator = cycle(self.train_loader)
        self.model.train()
        while self.step < self.lr_anneal_steps:
            batch = next(data_iterator)
            x = batch['image'].to(self.gpu_id)
            caption = None # batch['caption']
            loss = self._run_batch(batch, caption)
            print(f'gpu_id: {self.gpu_id}, step: {self.step}, loss: {loss} lr: {self.optimizer.param_groups[0]["lr"]}' )
            self.step += 1

    def _run_batch(self, x, c=None):
        self.optimizer.zero_grad()
        pred, target = self.model(x)
        loss = self.compute_loss(pred, target)
        self.backpropagation(loss)
        return loss.item()

    def backpropagation(self, loss):
        loss.backward()
        self.optimizer.step()
        self.anneal_lr()

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
