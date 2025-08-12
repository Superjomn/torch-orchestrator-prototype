import multiprocessing as mp

import torch
import torch.distributed as dist

from orchestrator import Orchestrator


def all_reduce_task(rank, world_size, results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    results.append(tensor.item())


def broadcast_task(rank, world_size, results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([123] if rank == 0 else [0], dtype=torch.float32, device=device)
    dist.broadcast(tensor, src=0)
    results.append(tensor.item())


def test_all_reduce():
    manager = mp.Manager()
    results = manager.list()
    orchestrator = Orchestrator(num_processes=2)
    orchestrator.submit(all_reduce_task, results)
    assert len(results) == 2
    assert all(value == 3.0 for value in results)


def test_broadcast():
    manager = mp.Manager()
    results = manager.list()
    orchestrator = Orchestrator(num_processes=2)
    orchestrator.submit(broadcast_task, results)
    assert len(results) == 2
    assert all(value == 123.0 for value in results)
