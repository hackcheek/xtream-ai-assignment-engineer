from kfp import Client, local, dsl

from challenge2.src.components.ingest import ingest_csv_component
from challenge2.src.components.preprocess import preprocess_csv_component
from challenge2.src.components.train import pytorch_model_train_component


local.init(runner=local.DockerRunner())

@dsl.pipeline(
    name='Challenge 2 pipeline',
    description='The goal is run this pipeline every time that database is updated and \
    train the model with the new data'
)
def ch2_pipeline(csv_file: str):
    user_data_path = ingest_csv_component(user_csv=csv_file)

    print(f'{user_data_path.output=}')
    # datasets = preprocess_csv_component(
    #     # user_data_path=user_data_path.output
    #     user_data_path=csv_file
    # )

    # print("OUTPUTS >>", datasets.outputs)
    # train_set_path = datasets.outputs.get(0)
    # test_set_path = datasets.outputs.get(1)
    # val_set_path = datasets.outputs.get(2)

    # best_model_weights_path = pytorch_model_train_component(
    #     train_dataset_path='',
    #     val_dataset_path=''
    # )

    # pytorch_model_test_component(
    #     best_model_weights_path,
    #     test_set_path
    # )


if __name__ == "__main__":
    # compiler.Compiler().compile(ch2_pipeline, 'challenge2/pipeline.yaml')
    client = Client()
    client.create_run_from_pipeline_func(
        ch2_pipeline,
        arguments={
            'csv_file': 'datasets/diamonds/new_records.csv',
        },
        enable_caching=False
    )
