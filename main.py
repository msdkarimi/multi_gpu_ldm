import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import image_transform, prepare_dataloader
from config.train_model_config import BATCH_SIZE, LR_INIT, get_unet_config
from data.data_loader import DareDataset
import os
from model.openai_model import UNetModel
import torch
from model.ldm import LDM
from trainer import Trainer


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)


def main(rank, world_size):
    ddp_setup(rank, world_size)
    train_dataset = DareDataset('/home/dare/ddpm/DDP-model-for-generating-conditional-images/data', 'train', transform=image_transform)
    val_dataset = DareDataset('/home/dare/ddpm/DDP-model-for-generating-conditional-images/data', 'validation', transform=image_transform)

    train_loader = prepare_dataloader(train_dataset, BATCH_SIZE)
    val_loader = prepare_dataloader(val_dataset, BATCH_SIZE)
    model = UNetModel(**get_unet_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, gpu_id=rank)
    trainer.start_training()
    destroy_process_group()


if __name__ == '__main__':
    _world_size = torch.cuda.device_count()
    mp.spawn(main, args=(_world_size,), nprocs=_world_size)
