import os
import torch
import torch.distributed as dist

backend = 'gloo'
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
dist.init_process_group(backend, rank=rank, world_size=world_size)

print(f"I am rank: {rank} out of a world size of {world_size}")
