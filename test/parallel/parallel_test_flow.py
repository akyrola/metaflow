from metaflow import FlowSpec, step, batch, current, parallel, Parameter, conda_base


@conda_base(python="3.6.0")
class ParallelTest(FlowSpec):
    """
    Test flow to test @parallel.
    """

    num_parallel = Parameter(
        "num_parallel", help="Number of nodes in cluster", default=3
    )

    @step
    def start(self):
        import sys

        print("Start", sys.version)
        self.next(self.parallel_step, num_parallel=self.num_parallel)

    @parallel
    # @batch
    @step
    def parallel_step(self):
        import sys

        print("PYTHON VERSION", sys.version)
        self.node_index = current.parallel.node_index
        self.num_nodes = current.parallel.num_nodes
        print("parallel_step: node {} finishing.".format(self.node_index))
        self.next(self.multinode_end)

    @step
    def multinode_end(self, inputs):
        j = 0
        for input in inputs:
            print(input)
        for input in inputs:
            print(input)
            assert input.node_index == j
            assert input.num_nodes == self.num_parallel
            j += 1
        assert j == self.num_parallel
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ParallelTest()
