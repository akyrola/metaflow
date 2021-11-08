import os

from metaflow.decorators import StepDecorator
from metaflow.unbounded_foreach import UnboundedForeachInput


class MultinodeDecorator(StepDecorator):
    name = "multinode"
    defaults = {
        "nodes": 2,
        "num_local_processes": 0,
        "framework": None,
    }

    def __init__(self, attributes=None, statically_defined=False):
        self.nodes = attributes["nodes"]
        self.framework = attributes["framework"]
        self.local_processes = attributes["num_local_processes"]
        super(MultinodeDecorator, self).__init__(attributes, statically_defined)

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        print("Multinode-decorator step-init: {}".format(self.framework))
        if self.framework == "pytorch":
            self._setup_pytorch()
        elif self.framework == "tensorflow":
            assert False, "tensorflow not implemented yet"
        elif self.framework == "custom":
            pass
        else:
            assert False, "Not support multinode framework: {}".format(self.framework)

    def _setup_pytorch(self):
        import torch

        num_local_processes = (
            self.local_processes if self.local_processes else torch.cuda.device_count()
        )
        if num_local_processes == 0:
            num_local_processes = 1

        assert (
            "MF_MULTINODE_NODE_INDEX" in os.environ
        ), "Multinode environment not configured by runtime!"

        print(
            "Configure pytorch environment. Number of local processes: {}, number of nodes: {}".format(
                num_local_processes, os.getenv("MF_MULTINODE_NUM_NODES", "1")
            )
        )
        # Torch's distributed settings
        os.environ["MASTER_PORT"] = "64398"  # arbitrary
        os.environ["MASTER_ADDR"] = os.getenv("MF_MULTINODE_MAIN_IP", "127.0.0.1")
        os.environ["NODE_RANK"] = os.getenv("MF_MULTINODE_NODE_INDEX", "0")
        os.environ["WORLD_SIZE"] = str(
            num_local_processes * int(os.getenv("MF_MULTINODE_NUM_NODES", "1"))
        )
        os.environ["NUM_NODES"] = os.getenv("MF_MULTINODE_NUM_NODES", "1")
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
