from challenge2.src.local.components.base import Component
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import PreprocessInput, PreprocessOutput
from typing import cast


class PreprocessComponent(Component):
    name = 'preprocess'

    @classmethod
    def do_preprocess(cls, input: PreprocessInput, ctx: Execution) -> PreprocessOutput:
        raise NotImplementedError

    @classmethod
    def preprocess(cls, input: PreprocessInput) -> PreprocessOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_preprocess(input, cast(Execution, ctx))
