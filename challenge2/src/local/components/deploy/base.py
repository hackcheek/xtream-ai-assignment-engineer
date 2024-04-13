from challenge2.src.local.components.base import Component
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import DeployInput, DeployOutput
from typing import cast


class DeployComponent(Component):
    name = 'deploy'

    @classmethod
    def do_deploy(cls, input: DeployInput, ctx: Execution) -> DeployOutput:
        raise NotImplementedError

    @classmethod
    def deploy(cls, input: DeployInput) -> DeployOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_deploy(input, cast(Execution, ctx))
