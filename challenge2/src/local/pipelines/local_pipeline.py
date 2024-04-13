from challenge2.src.configs import CH2PipelineConfig
from challenge2.src.local.components.deploy.ch2 import CH2Deploy
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
    CH2Deploy
):
    def __init__(self, input: CH2PipelineInput, cfg = CH2PipelineConfig()):
        self.input = input
        self.cfg = cfg
        self.__class__.cfg = cfg
