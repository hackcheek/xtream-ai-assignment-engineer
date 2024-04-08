from kfp import dsl


@dsl.component(base_image='python:3.8')
def ingest_csv_component(user_csv: str) -> str:
    import pandas as pd
    """
    Note that, in a real world environment, this kind of component ingest data to a database and would return table name.
    However for this challenge I'll ingest the data to a temp csv file and return the path.
    """
    print("[*] Ingesting data")
    output_path = "challenge2/datasets/user_data.csv"
    user_data = pd.read_csv(user_csv)
    user_data.to_csv(output_path)
    return output_path
