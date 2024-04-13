from tempfile import NamedTemporaryFile
import pandas as pd
import shutil

from challenge2.src.local.components.ingest.base import IngestComponent
from challenge2.src.utils.metadata.artifacts.dataset import DatasetArtifact
from challenge2.src.utils.metadata.artifacts.model import ModelArtifact
from challenge2.src.utils.schemas.pipeline import CH2PipelineInput, IngestOutput


class CH2Ingest(IngestComponent):
    
    @staticmethod
    def _mv_to_work_path(path: str) -> str:
        new_path = NamedTemporaryFile().name
        shutil.move(path, new_path)
        return new_path

    @classmethod
    def do_ingest(cls, input: CH2PipelineInput) -> IngestOutput:
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
        
        

def _ingest_csv_component(
    csv_loc: str,
    baseline_model_loc: str,
    baseline_data_loc: str,
    port: int,
    access_key: str,
    secret_key: str,
    user_data: Output[Dataset],
    base_data: Output[Dataset],
    base_model: Output[Model]
):
    import pandas as pd
    from io import BytesIO


    print("[*] Ingesting data")

    response = client.get_object(*csv_loc.split('/', 1))
    data = pd.read_csv(BytesIO(response.data))
    data.to_csv(user_data.path, index=False)

    response = client.get_object(*baseline_data_loc.split('/', 1))
    _base_data = pd.read_csv(BytesIO(response.data))
    _base_data.to_csv(base_data.path, index=False)

    response = client.get_object(*baseline_model_loc.split('/', 1))
    client.fget_object(*baseline_model_loc.split('/', 1), base_model.path)


ingest_csv_component = partial(
    _ingest_csv_component,
    port=MinioConfig.PORT,
    access_key=MinioConfig.ACCESS_KEY,
    secret_key=MinioConfig.SECRET_KEY
)
