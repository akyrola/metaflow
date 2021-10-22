
from metaflow.unbounded_foreach import UBF_CONTROL, UBF_TASK
import sys
from metaflow.decorators import StepDecorator
from metaflow.util import to_unicode
from metaflow.cli_args import cli_args
import subprocess

class MultinodeDecorator(StepDecorator):
    name = 'multinode'
    defaults = {
        'nodes': 2,
    }

    def __init__(self, attributes=None, statically_defined=False):
        self.nodes = attributes["nodes"]
        super(MultinodeDecorator, self).__init__(attributes, statically_defined)
