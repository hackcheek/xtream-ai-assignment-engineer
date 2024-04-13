from challenge2.src.local.components.evaluation.base import EvaluationComponent
from challenge2.src.utils.metadata.executions.base import Execution
from challenge2.src.utils.schemas.pipeline import EvaluationInput, EvaluationOutput


class CH2Evaluation(EvaluationComponent):
    @classmethod
    def do_eval(cls, input: EvaluationInput, ctx: Execution) -> EvaluationOutput:
        
