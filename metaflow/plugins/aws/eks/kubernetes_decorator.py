import os
import sys
import platform
import requests

from metaflow import current, util
from metaflow.decorators import StepDecorator
from metaflow.metadata import MetaDatum
from metaflow.metadata.util import sync_local_metadata_to_datastore
from metaflow.metaflow_config import (
    KUBERNETES_CONTAINER_IMAGE,
    KUBERNETES_CONTAINER_REGISTRY,
    DATASTORE_LOCAL_DIR,
)
from metaflow.plugins import ResourcesDecorator
from metaflow.plugins.timeout_decorator import get_run_time_limit_for_task
from metaflow.sidecar import SidecarSubProcess

from .kubernetes import KubernetesException
from ..aws_utils import get_docker_registry
from metaflow.unbounded_foreach import UBF_CONTROL


class KubernetesDecorator(StepDecorator):
    """
    TODO (savin): Update this docstring.
    Step decorator to specify that this step should execute on Kubernetes.

    This decorator indicates that your step should execute on Kubernetes. Note
    that you can apply this decorator automatically to all steps using the
    ```--with kubernetes``` argument when calling run/resume. Step level
    decorators within the code are overrides and will force a step to execute
    on Kubernetes regardless of the ```--with``` specification.

    To use, annotate your step as follows:
    ```
    @kubernetes
    @step
    def my_step(self):
        ...
    ```
    Parameters
    ----------
    cpu : int
        Number of CPUs required for this step. Defaults to 1. If @resources is
        also present, the maximum value from all decorators is used
    gpu : int
        Number of GPUs required for this step. Defaults to 0. If @resources is
        also present, the maximum value from all decorators is used
    memory : int
        Memory size (in MB) required for this step. Defaults to 4096. If
        @resources is also present, the maximum value from all decorators is
        used
    image : string
        Docker image to use when launching on Kubernetes. If not specified, a
        default docker image mapping to the current version of Python is used
    shared_memory : int
        The value for the size (in MiB) of the /dev/shm volume for this step.
        This parameter maps to the --shm-size option to docker run.
    """

    name = "kubernetes"
    defaults = {
        "cpu": "1",
        "memory": "4096",
        "disk": "10240",
        "image": None,
        "service_account": None,
        "secrets": None,  # e.g., mysecret
        "node_selector": None,  # e.g., kubernetes.io/os=linux
        "gpu": "0",
        # "shared_memory": None,
        "namespace": None,
    }
    package_url = None
    package_sha = None
    run_time_limit = None

    def __init__(self, attributes=None, statically_defined=False):
        super(KubernetesDecorator, self).__init__(attributes, statically_defined)

        # TODO: Unify the logic with AWS Batch
        # If no docker image is explicitly specified, impute a default image.
        if not self.attributes["image"]:
            # If metaflow-config specifies a docker image, just use that.
            if KUBERNETES_CONTAINER_IMAGE:
                self.attributes["image"] = KUBERNETES_CONTAINER_IMAGE
            # If metaflow-config doesn't specify a docker image, assign a
            # default docker image.
            else:
                # Default to vanilla Python image corresponding to major.minor
                # version of the Python interpreter launching the flow.
                self.attributes["image"] = "python:%s.%s" % (
                    platform.python_version_tuple()[0],
                    platform.python_version_tuple()[1],
                )
        # Assign docker registry URL for the image.
        if not get_docker_registry(self.attributes["image"]):
            if KUBERNETES_CONTAINER_REGISTRY:
                self.attributes["image"] = "%s/%s" % (
                    KUBERNETES_CONTAINER_REGISTRY.rstrip("/"),
                    self.attributes["image"],
                )

    # Refer https://github.com/Netflix/metaflow/blob/master/docs/lifecycle.png
    # to understand where these functions are invoked in the lifecycle of a
    # Metaflow flow.
    def step_init(self, flow, graph, step, decos, environment, flow_datastore, logger):
        # Executing Kubernetes jobs requires a non-local datastore at the
        # moment.
        # TODO: To support MiniKube we need to enable local datastore execution.
        if flow_datastore.TYPE != "s3":
            raise KubernetesException(
                "The *@kubernetes* decorator requires --datastore=s3 " "at the moment."
            )

        # Set internal state.
        self.logger = logger
        self.environment = environment
        self.step = step
        self.flow_datastore = flow_datastore
        for deco in decos:
            if isinstance(deco, ResourcesDecorator):
                for k, v in deco.attributes.items():
                    # We use the larger of @resources and @k8s attributes
                    # TODO: Fix https://github.com/Netflix/metaflow/issues/467
                    my_val = self.attributes.get(k)
                    if not (my_val is None and v is None):
                        self.attributes[k] = str(max(int(my_val or 0), int(v or 0)))

        # Set run time limit for the Kubernetes job.
        self.run_time_limit = get_run_time_limit_for_task(decos)
        if self.run_time_limit < 60:
            raise KubernetesException(
                "The timeout for step *{step}* should be "
                "at least 60 seconds for execution on "
                "Kubernetes.".format(step=step)
            )

    def runtime_init(self, flow, graph, package, run_id):
        # Set some more internal state.
        self.flow = flow
        self.graph = graph
        self.package = package
        self.run_id = run_id

    def runtime_task_created(
        self, task_datastore, task_id, split_index, input_paths, is_cloned, ubf_context
    ):
        # To execute the Kubernetes job, the job container needs to have
        # access to the code package. We store the package in the datastore
        # which the pod is able to download as part of it's entrypoint.
        if not is_cloned:
            self._save_package_once(self.flow_datastore, self.package)

    def runtime_step_cli(
        self, cli_args, retry_count, max_user_code_retries, ubf_context
    ):
        if retry_count <= max_user_code_retries:
            # After all attempts to run the user code have failed, we don't need
            # to execute on Kubernetes anymore. We can execute possible fallback
            # code locally.
            cli_args.commands = ["kubernetes", "step"]
            cli_args.command_args.append(self.package_sha)
            cli_args.command_args.append(self.package_url)

            # --namespace is used to specify Metaflow namespace (different
            # concept from k8s namespace).
            for k, v in self.attributes.items():
                if k == "namespace":
                    cli_args.command_options["k8s_namespace"] = v
                else:
                    cli_args.command_options[k] = v
            cli_args.command_options["run-time-limit"] = self.run_time_limit
            cli_args.entrypoint[0] = sys.executable

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
        max_retries,
        ubf_context,
        inputs,
    ):
        self.metadata = metadata
        self.task_datastore = task_datastore

        # task_pre_step may run locally if fallback is activated for @catch
        # decorator. In that scenario, we skip collecting Kubernetes execution
        # metadata. A rudimentary way to detect non-local execution is to
        # check for the existence of METAFLOW_KUBERNETES_WORKLOAD environment
        # variable.

        if "METAFLOW_KUBERNETES_WORKLOAD" in os.environ:
            meta = {}
            # TODO: Get kubernetes job id and job name
            meta["kubernetes-pod-id"] = os.environ["METAFLOW_KUBERNETES_POD_ID"]
            meta["kubernetes-pod-name"] = os.environ["METAFLOW_KUBERNETES_POD_NAME"]
            meta["kubernetes-pod-namespace"] = os.environ[
                "METAFLOW_KUBERNETES_POD_NAMESPACE"
            ]
            # meta['kubernetes-job-attempt'] = ?

            entries = [
                MetaDatum(field=k, value=v, type=k, tags=[]) for k, v in meta.items()
            ]
            # Register book-keeping metadata for debugging.
            metadata.register_metadata(run_id, step_name, task_id, entries)

            # Start MFLog sidecar to collect task logs.
            self._save_logs_sidecar = SidecarSubProcess("save_logs_periodically")

        num_parallel = int(os.environ.get("MF_KUBERNETES_NUM_PARALLEL", 1))
        if num_parallel > 1 and ubf_context == UBF_CONTROL:
            # UBF handling for multinode case
            control_task_id = current.task_id
            top_task_id = control_task_id.replace("control-", "")  # chop "-0"
            mapper_task_ids = [control_task_id] + [
                "%s-node-%d" % (top_task_id, node_idx)
                for node_idx in range(1, num_parallel)
            ]
            flow._control_mapper_tasks = [
                "%s/%s/%s" % (run_id, step_name, mapper_task_id)
                for mapper_task_id in mapper_task_ids
            ]
            flow._control_task_is_mapper_zero = True

        if num_parallel > 1:
            _setup_multinode_environment()

    def task_post_step(
        self, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        # task_post_step may run locally if fallback is activated for @catch
        # decorator.
        if "METAFLOW_KUBERNETES_WORKLOAD" in os.environ:
            # If `local` metadata is configured, we would need to copy task
            # execution metadata from the AWS Batch container to user's
            # local file system after the user code has finished execution.
            # This happens via datastore as a communication bridge.
            if self.metadata.TYPE == "local":
                # Note that the datastore is *always* Amazon S3 (see
                # runtime_task_created function).
                sync_local_metadata_to_datastore(
                    DATASTORE_LOCAL_DIR, self.task_datastore
                )

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        # task_exception may run locally if fallback is activated for @catch
        # decorator.
        if "METAFLOW_KUBERNETES_WORKLOAD" in os.environ:
            # If `local` metadata is configured, we would need to copy task
            # execution metadata from the AWS Batch container to user's
            # local file system after the user code has finished execution.
            # This happens via datastore as a communication bridge.
            if self.metadata.TYPE == "local":
                # Note that the datastore is *always* Amazon S3 (see
                # runtime_task_created function).
                sync_local_metadata_to_datastore(
                    DATASTORE_LOCAL_DIR, self.task_datastore
                )

    def task_finished(
        self, step_name, flow, graph, is_task_ok, retry_count, max_retries
    ):
        try:
            self._save_logs_sidecar.kill()
        except:
            # Best effort kill
            pass

    @classmethod
    def _save_package_once(cls, flow_datastore, package):
        if cls.package_url is None:
            cls.package_url, cls.package_sha = flow_datastore.save_data(
                [package.blob], len_hint=1
            )[0]


def _setup_multinode_environment():
    # setup the multinode environment variables.
    main_ip = open("/etc/volcano/m.host", "r").read()
    print("MAIN IP", main_ip)
    os.environ["MF_PARALLEL_MAIN_IP"] = main_ip
    os.environ["MF_PARALLEL_NUM_NODES"] = os.environ["MF_KUBERNETES_NUM_PARALLEL"]
    if os.environ.get("MF_KUBERNETES_ROLE", "main") == "secondary":
        os.environ["MF_PARALLEL_NODE_INDEX"] = str(1 + int(os.environ["VK_TASK_INDEX"]))
    else:
        os.environ["MF_PARALLEL_NODE_INDEX"] = os.environ["VK_TASK_INDEX"]
