from challenge2.src.local.components.base import Component
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import EvaluationOutput, EvaluationInput
from typing import cast


class EvaluationComponent(Component):
    name = 'evaluation'

    @classmethod
    def do_eval(cls, input: EvaluationInput, ctx: Execution) -> EvaluationOutput:
        raise NotImplementedError

    @classmethod
    def eval(cls, input: EvaluationInput) -> EvaluationOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_eval(input, cast(Execution, ctx))
