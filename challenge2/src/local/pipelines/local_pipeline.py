from challenge2.src.configs import CH2PipelineConfig
from challenge2.src.local.components.evaluation.ch2 import CH2Evaluation
from challenge2.src.local.components.ingest.ch2 import CH2Ingest
from challenge2.src.local.components.preprocess.ch2 import CH2Preprocess
from challenge2.src.local.components.train.ch2 import CH2Train
from challenge2.src.local.pipelines.base import BaseReTrainPipeline
from challenge2.src.utils.schemas.pipeline import CH2PipelineInput


class CH2LocalPipeline(
    BaseReTrainPipeline,
    CH2Ingest,
    CH2Preprocess,
    CH2Train,
    CH2Evaluation,
)
    def __init__(self, input: CH2PipelineInput, cfg = CH2PipelineConfig()):
        self.input = input
        self.cfg = cfg


if __name__ == "__main__":
    input = CH2PipelineInput(
        base_data_path = 'datasets/diamonds/diamonds.csv',
        user_data_path = 'datasets/diamonds/new_records.csv',
        base_model_path = 'challenge1/base_model.pt',
    )
    CH2LocalPipeline(input).run()
