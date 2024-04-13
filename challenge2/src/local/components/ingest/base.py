from challenge2.src.local.components.base import Component
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import PipelineInput, IngestOutput
from typing import cast


class IngestComponent(Component):
    name = 'ingest'

    @classmethod
    def do_ingest(cls, input: PipelineInput, ctx: Execution) -> IngestOutput:
        raise NotImplementedError

    @classmethod
    def ingest(cls, input: PipelineInput) -> IngestOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_ingest(input, cast(Execution, ctx))
