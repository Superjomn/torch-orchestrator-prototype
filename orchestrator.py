import os
import socket
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


class Orchestrator:
    """Simple process orchestrator using torch.distributed."""

    def __init__(self, num_processes, node_rank=0, num_nodes=1,
                 master_addr=None, master_port=29500, backend=None):
        self.num_processes = num_processes
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.master_addr = master_addr or socket.gethostname()
        self.master_port = master_port
        self.backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")

    def submit(self, task, *args, **kwargs):
        """Run ``task`` across all processes."""
        # We can insert some RPC in the processes to support more flexible features
        mp.spawn(self._worker, nprocs=self.num_processes,
                 args=(task, args, kwargs))

    def _worker(self, local_rank, task, args, kwargs):
        global_rank = self.node_rank * self.num_processes + local_rank
        world_size = self.num_nodes * self.num_processes

        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.master_port))
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        print(f"Init rank {global_rank} of {world_size} processes, communicating to master: {self.master_addr}:{self.master_port}")

        if self.backend == "nccl":
            torch.cuda.set_device(local_rank)

        dist.init_process_group(self.backend, rank=global_rank,
                                world_size=world_size)
        try:
            task(global_rank, world_size, *args, **kwargs)
        finally:
            dist.destroy_process_group()
