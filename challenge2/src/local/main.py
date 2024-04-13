from challenge2.src.configs import CH2PipelineConfig
from challenge2.src.local.pipelines import CH2LocalPipeline
from challenge2.src.utils.schemas.pipeline import CH2PipelineInput


if __name__ == "__main__":
    input = CH2PipelineInput(
        base_data_path = 'datasets/diamonds/diamonds.csv',
        user_data_path = 'datasets/diamonds/new_records.csv',
        base_model_path = 'challenge1/best_model.pt',
    )
    CH2LocalPipeline(input, CH2PipelineConfig(num_epochs=5)).run()
