import shutil
import pandas as pd

from challenge2.src.configs import CH2PipelineConfig
from challenge2.src.local.components.deploy.base import DeployComponent
from challenge2.src.utils.metadata.artifacts.model import ModelArtifact
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import DeployInput, DeployOutput


class CH2Deploy(DeployComponent):
    cfg: CH2PipelineConfig

    @classmethod
    def do_deploy(cls, input: DeployInput, ctx: Execution) -> DeployOutput:
        try:
            model_op = ModelArtifact('trained_model', uri=cls.cfg.output_model_path)
            data_op = ModelArtifact('data', uri=cls.cfg.output_data_path)

            _user_data = pd.read_csv(input.user_data.uri)
            _base_data = pd.read_csv(input.base_data.uri)

            new_baseline_data = pd.concat((_base_data, _user_data)) 

            new_baseline_data.to_csv(data_op.uri)
            shutil.copy(input.model.uri, model_op.uri)
        except Exception as err:
            ctx.update_metadata({'error_in_deploy': err}) 
            return DeployOutput(False)

        return DeployOutput(True)
