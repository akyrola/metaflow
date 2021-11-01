import inspect
import subprocess
import pickle
import tempfile
import os
import torch
import sys
from . import paths


class PyTorchHelper:
    @staticmethod
    def run_trainer(
        run, target, num_local_workers=max(torch.cuda.device_count(), 1), **kwargs
    ):
        # Inject checkpoint args
        sig = inspect.signature(target)
        if "checkpoint_url" in sig.parameters:
            kwargs["checkpoint_url"] = paths.get_s3_checkpoint_url(run)
        else:
            print(
                "NOTE: checkpoint_url not an argument to the pytorch target '{}'".format(
                    target.__name__
                )
            )
        if "latest_checkpoint_url" in sig.parameters:
            assert (
                "checkpoint_url" in sig.parameters
            ), "With latest_checkpoint_url argument, need also add checkpoint_url argument."
            kwargs["latest_checkpoint_url"] = paths.get_s3_latest_checkpoint_url(run)
        else:
            print(
                "NOTE: latest_checkpoint_url not an argument to the pytorch target '{}'".format(
                    target.__name__
                )
            )
        if "logger_url" in sig.parameters:
            kwargs["logger_url"] = paths.get_s3_logger_url(run=run)
        else:
            print("NOTE: logger_url not an argument to the pytorch target '{}").format(
                target.__name__
            )

        # SETUP DISTRIBUTED ENV FOR METAFLOW & TORCH
        setup_torch_distributed(num_local_workers)

        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as args_file:
            pickle.dump(kwargs, file=args_file)

        # Run the target via "spawn" command of the flow.
        subprocess.run(
            check=True,
            args=[
                sys.executable,
                sys.argv[0],
                "spawn",
                target.__module__,
                target.__name__,
                args_file.name,
            ],
        )


def setup_torch_distributed(num_local_devices):
    """
    Set up environment variables for PyTorch's distributed (DDP).
    """
    os.environ["MASTER_PORT"] = "64398"  # arbitrary
    os.environ["MASTER_ADDR"] = os.environ.get("MF_MULTINODE_MAIN_IP", "127.0.0.1")
    os.environ["NODE_RANK"] = os.environ.get("MF_MULTINODE_MODE_INDEX", "0")
    os.environ["WORLD_SIZE"] = str(
        int(os.environ.get("MF_MULTINODE_NUM_NODES", "1")) * num_local_devices
    )
    os.environ["NUM_NODES"] = os.environ.get("MF_MULTINODE_NUM_NODES", "1")
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # NCCL crashes on aws batch!
