from tempfile import NamedTemporaryFile
import pandas as pd
import shutil

from challenge2.src.local.components.ingest.base import IngestComponent
from challenge2.src.utils.metadata.artifacts.dataset import DatasetArtifact
from challenge2.src.utils.metadata.artifacts.model import ModelArtifact
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import CH2PipelineInput, IngestOutput


class CH2Ingest(IngestComponent):
    
    @staticmethod
    def _mv_to_work_path(path: str) -> str:
        new_path = NamedTemporaryFile().name
        shutil.copy(path, new_path)
        return new_path

    @classmethod
    def do_ingest(cls, input: CH2PipelineInput, ctx: Execution) -> IngestOutput:
        _base_model = ModelArtifact(
            'base_model',
            cls._mv_to_work_path(input.base_model_path)
        )
        _base_data = DatasetArtifact(
            'base_data',
            cls._mv_to_work_path(input.base_data_path)
        )
        _user_data = DatasetArtifact(
            'user_data',
            cls._mv_to_work_path(input.user_data_path)
        )

        return IngestOutput(
            base_model=_base_model,
            base_data=_base_data,
            user_data=_user_data,
        ) 
