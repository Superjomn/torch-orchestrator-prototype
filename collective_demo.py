import os
import socket
import torch
import torch.distributed as dist

from orchestrator import Orchestrator


def all_reduce_task(rank: int, world_size: int) -> None:
    """Simple task that performs an all_reduce across ranks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}/{world_size} performed all_reduce, got tensor {tensor.item()}")


def main():
    num_processes = int(os.environ.get("NUM_PROCESSES", 2))
    num_nodes = int(
        os.environ.get("NUM_NODES")
        or os.environ.get("SLURM_JOB_NUM_NODES", 1)
    )
    node_rank = int(
        os.environ.get("NODE_RANK")
        or os.environ.get("SLURM_NODEID", 0)
    )
    master_addr = os.environ.get("MASTER_ADDR", socket.gethostname())
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    master = Orchestrator(
        num_processes=num_processes,
        node_rank=node_rank,
        num_nodes=num_nodes,
        master_addr=master_addr,
        master_port=master_port,
    )

    master.submit(all_reduce_task)


if __name__ == "__main__":
    main()
