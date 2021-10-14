import os
import socket
from datetime import timedelta
"""
Utilities for parallel computation in the AWS batch environment.
"""

class BatchParallel:
    def __init__(self, local_rank, num_local_workers):
        self.local_rank = local_rank
        self.num_local_workers = num_local_workers
        assert self.local_rank < self.num_local_workers
        print_worker_info(self.local_rank, self.num_local_workers)
        self._initialized = False

    def global_rank(self):
        assert self._initialized, "Call torch_dist_init() first!"
        return get_global_rank(self.local_rank, self.num_local_workers)

    def torch_validate_setup(self):
        assert self._initialized
        torch_validate_distributed_setup(self.local_rank, self.num_local_workers)

    def is_main_worker(self):
        return is_main_rank(self.local_rank)

    def torch_dist_init(self):
        import torch.distributed as dist
        dist.init_process_group(
            "gloo",
            rank=get_global_rank(self.local_rank, self.num_local_workers),
            world_size=get_world_size(self.num_local_workers),
            init_method="tcp://{}:65434".format(get_main_node_ip()),  # TODO: get port from config?
            timeout=timedelta(seconds=300)
        )
        self._initialized = True


def setup_torch_distributed(num_local_devices):
    os.environ["MASTER_PORT"] = "64398"  # arbitrary
    os.environ["MASTER_ADDR"] = str(get_main_node_ip())
    os.environ["NODE_RANK"] = str(get_node_index())
    os.environ["WORLD_SIZE"] = str(get_world_size(num_local_devices))
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["METAFLOW_SHADOW_TASK"] = "1"

def get_number_of_nodes():
    return int(os.getenv('AWS_BATCH_JOB_NUM_NODES', '1'))


def get_node_index():
    return int(os.getenv('AWS_BATCH_JOB_NODE_INDEX', '0'))


def get_main_node_index():
    return int(os.getenv('AWS_BATCH_JOB_MAIN_NODE_INDEX', '0'))


def is_main_rank(local_rank):
    return local_rank == 0 and get_main_node_index() == get_node_index()


def is_main_node():
    return get_main_node_index() == get_node_index()


def get_world_size(num_local_workers):
    """
    Number of total workers that need to rendezvous.
    @param num_local_workers:  number of local workers per node. Must be same on all nodes.
    """
    assert num_local_workers >= 1, "Number of local workers must be at least 1"
    return get_number_of_nodes() * num_local_workers


def get_main_node_ip():
    if get_number_of_nodes() == 1:
        # Just use localhost as not a multi-node env
        print("Only single node, use localhost for rendezvous")
        return "127.0.0.1"
    if is_main_node():
        local_ips = socket.gethostbyname_ex(socket.gethostname())[-1]
        assert local_ips, "Could not find local ip address"
        main_addr = local_ips[0]
    else:
        main_addr = os.getenv('AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS')
        assert main_addr, "Could not find main node IP address!"
    return main_addr


def get_global_rank(local_rank, num_local_workers):
    return local_rank + num_local_workers * get_node_index()


def print_worker_info(local_rank, workers_per_node):
    print("AWS Batch Parallel configuration:")
    print("- number of distributed nodes:", get_number_of_nodes())
    print("- this worker node index:", get_node_index())
    print("- local worker rank:", local_rank)
    print("- global rank:", get_global_rank(local_rank, workers_per_node))
    print("- world size:", get_world_size(workers_per_node))
    print("- main node IP:", get_main_node_ip())


def torch_validate_distributed_setup(local_rank, num_local_workers):
    import torch
    import torch.distributed as dist
    print("Going to validate the setup")
    all_ranks = [torch.zeros(1) for _ in range(get_world_size(num_local_workers))]
    my_rank = torch.zeros(1) + get_global_rank(local_rank, num_local_workers)
    dist.all_gather(all_ranks, my_rank)
    for j in range(len(all_ranks)):
        assert int(all_ranks[j][0]) == j, "Not all ranks had same number of local workers?"
