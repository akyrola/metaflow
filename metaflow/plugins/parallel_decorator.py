from metaflow.decorators import StepDecorator
from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.exception import MetaflowException
import os
import sys


class ParallelDecorator(StepDecorator):
    name = "parallel"
    defaults = {}
    IS_PARALLEL = True

    def __init__(self, attributes=None, statically_defined=False):
        super(ParallelDecorator, self).__init__(attributes, statically_defined)

    def runtime_step_cli(
        self, cli_args, retry_count, max_user_code_retries, ubf_context
    ):

        if ubf_context == UBF_CONTROL:
            num_parallel = cli_args.task.ubf_iter.num_parallel
            cli_args.command_options["num-parallel"] = str(num_parallel)

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        self.environment = environment

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        if (
            ubf_context == UBF_CONTROL
            and os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local"
        ):
            from functools import partial

            return partial(
                _local_multinode_control_task_step_func,
                flow,
                self.environment,
                step_func,
                retry_count,
            )
        else:
            return step_func


def _local_multinode_control_task_step_func(flow, env_to_use, step_func, retry_count):
    """
    Used as multinode UBF control task when run in local mode.
    """
    from metaflow import current
    from metaflow.cli_args import cli_args
    from metaflow.unbounded_foreach import UBF_TASK
    import subprocess

    print("UBF CONTROL:", sys.version)

    assert flow._unbounded_foreach
    foreach_iter = flow._parallel_ubf_iter
    if foreach_iter.__class__.__name__ != "ParallelUBF":
        raise MetaflowException(
            "Expected ParallelUBFIter iterator object, got:"
            + foreach_iter.__class__.__name__
        )

    num_parallel = foreach_iter.num_parallel
    os.environ["MF_PARALLEL_NUM_NODES"] = str(num_parallel)
    os.environ["MF_PARALLEL_MAIN_IP"] = "127.0.0.1"

    run_id = current.run_id
    step_name = current.step_name
    control_task_id = current.task_id

    (_, split_step_name, split_task_id) = control_task_id.split("-")[1:]
    # UBF handling for multinode case
    top_task_id = control_task_id.replace("control-", "")  # chop "-0"
    mapper_task_ids = []

    env_to_use = getattr(env_to_use, "base_env", env_to_use)
    executable = env_to_use.executable(step_name)
    script = sys.argv[0]

    # start workers
    subprocesses = []
    for node_index in range(0, num_parallel):
        task_id = "%s-node-%d" % (top_task_id, node_index)
        mapper_task_ids.append(task_id)
        os.environ["MF_PARALLEL_NODE_INDEX"] = str(node_index)
        input_paths = "%s/%s/%s" % (run_id, split_step_name, split_task_id)
        # Override specific `step` kwargs.
        kwargs = cli_args.step_kwargs
        kwargs["split_index"] = str(node_index)
        kwargs["run_id"] = run_id
        kwargs["task_id"] = task_id
        kwargs["input_paths"] = input_paths
        kwargs["ubf_context"] = UBF_TASK
        kwargs["retry_count"] = str(retry_count)

        step_cmds = cli_args.step_command(
            executable, script, step_name, step_kwargs=kwargs
        )

        # Print cmdline for execution. Doesn't work without the temporary
        # unicode object while using `print`.
        print("Original executable", sys.executable)
        print(
            u"[${cwd}] Starting split#{split} with cmd:{cmd}".format(
                cwd=os.getcwd(), split=node_index, cmd=" ".join(step_cmds)
            )
        )
        p = subprocess.Popen(step_cmds)
        subprocesses.append(p)

    flow._control_mapper_tasks = [
        "%s/%s/%s" % (run_id, step_name, mapper_task_id)
        for mapper_task_id in mapper_task_ids
    ]
    # flow._control_task_is_mapper_zero = True
    # join the subprocesses
    print("Waiting for the subprocesses to finish")
    for p in subprocesses:
        p.wait()
        if p.returncode:
            raise Exception("Subprocess failed, return code {}".format(p.returncode))
    print("Done")
