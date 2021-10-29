import inspect
import subprocess
import pickle
import tempfile
import os
import torch
import sys
from metaflow import S3


class PyTorchHelper:
    def __init__(self, run):
        self._run = run

    def run(self, target, num_local_workers=max(torch.cuda.device_count(), 1), **kwargs):
        # Inject checkpoint args
        s3 = S3(run=self._run)
        sig = inspect.signature(target)
        if "checkpoint_url" in sig.parameters:
            kwargs["checkpoint_url"] = s3.checkpoint_url
        else:
            print("NOTE: checkpoint_url not an argument to the pytorch target '{}'".format(target.__name__))
        if "latest_checkpoint_url" in sig.parameters:
            kwargs["latest_checkpoint_url"] = s3.latest_checkpoint_url()
        else:
            print("NOTE: latest_checkpoint_url not an argument to the pytorch target '{}'".format(target.__name__))
        if "logger_url" in sig.parameters:
            kwargs["logger_url"] = s3.logger_url

        print("Calling pytorch target {}".format(target.__name__))
        print("  parameters: {}".format(kwargs))


        # SETUP DISTRIBUTED ENV FOR METAFLOW & TORCH
        setup_torch_distributed(num_local_workers)

        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as args_file:
            pickle.dump(kwargs, file=args_file)

        print([sys.executable, sys.argv[0], "spawn", target.__module__ , target.__name__, args_file.name])
        subprocess.run(check=True,  args=[sys.executable, sys.argv[0], "spawn", target.__module__ , target.__name__, args_file.name])

def setup_torch_distributed(num_local_devices):
    os.environ["MASTER_PORT"] = "64398"  # arbitrary
    os.environ["MASTER_ADDR"] = os.environ.get("MF_MULTINODE_MAIN_IP", "127.0.0.1")
    os.environ["NODE_RANK"] =  os.environ.get("MF_MULTINODE_MODE_INDEX", "0")
    os.environ["WORLD_SIZE"] = str(int(os.environ.get("MF_MULTINODE_NUM_NODES", "1")) * num_local_devices)
    os.environ["NUM_NODES"] = os.environ.get("MF_MULTINODE_NUM_NODES", "1")
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # NCCL crashes on aws batch!