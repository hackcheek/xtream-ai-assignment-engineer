from typing import Any
from challenge2.src.local.components.ingest.base import IngestComponent
from challenge2.src.local.components.preprocess.base import PreprocessComponent
from challenge2.src.local.components.train.base import TrainComponent
from challenge2.src.local.components.evaluation.base import EvaluationComponent
from challenge2.src.local.components.deploy.base import DeployComponent
from challenge2.src.utils.schemas.pipeline import DeployInput, EvaluationInput, PipelineInput, PipelineOutput, PipelineConfig, PreprocessInput, TrainInput


class Pipeline:
    def __init__(self, input: PipelineInput, cfg: PipelineConfig):
        self.cfg = cfg
        self.input = input

    def run(self) -> PipelineOutput | None:
        raise NotImplementedError


class BaseReTrainPipeline(
    Pipeline,
    IngestComponent,
    PreprocessComponent,
    TrainComponent,
    EvaluationComponent,
    DeployComponent,
):
    def run(self) -> None:
        ingest_op = self.ingest(self.input)

        preproc_in = PreprocessInput(ingest_op.base_data, ingest_op.user_data)
        preproc_op = self.preprocess(preproc_in)

        train_in = TrainInput(preproc_op.train_dataset, preproc_op.val_dataset)
        train_op = self.train(train_in)
        
        eval_in = EvaluationInput(train_op.model, ingest_op.base_model, preproc_op.test_dataset)
        eval_op = self.eval(eval_in)

        results_dict = eval_op.results.content()
        if results_dict['current_model_loss'] < results_dict['base_model_loss']:
            deploy_in = DeployInput(ingest_op.user_data, train_op.model)
            deploy_op = self.deploy(deploy_in)

            if not deploy_op.deployed:
                raise RuntimeError('Error trying to deploy')

    def build_input(self, attributes: dict[str, Any]):
        return type('', (object,), attributes)
