# Torch Distributed Orchestrator Prototype

This repository provides a minimal `torch.distributed` orchestration layer
for running collective operations across multiple processes. The
`Orchestrator` class spawns local workers and sets up the distributed
process group, allowing you to submit a task function that runs on every
rank.

## Requirements
- PyTorch with distributed support
- OpenMPI for `mpirun` or a Slurm cluster for `srun` (optional)

## Usage
### Python API
```python
from orchestrator import Orchestrator

master = Orchestrator(num_processes=8)
master.submit(task=my_callable)
```

### Single node
Run two processes on the local machine:
```bash
NUM_PROCESSES=2 python collective_demo.py
```

### Multi node with `srun`
Assuming two nodes with two GPUs each:
```bash
srun --nodes=2 --ntasks=2 \
     --export=ALL,NUM_PROCESSES=2,MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
     python collective_demo.py
```
Environment variables `SLURM_JOB_NUM_NODES` and `SLURM_NODEID` are used to
determine the total number of nodes and the current node's rank.

Each worker prints the summed tensor value after the `all_reduce` operation,
showing that communication occurred across all processes.

## Tests
Run the included unit tests to exercise the prototype collectives:

```bash
pytest tests/test_collectives.py
```
