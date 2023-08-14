from pathlib import Path

import pandas as pd
import typer
from sentence_transformers import SentenceTransformer, util


def main(
        input_model: str = typer.Option(default="dmis-lab/biobert-base-cased-v1.2",
                                         help="Path to the input model"),
        input_data_file: Path = typer.Option(default="../data/SUBJECT_ID_to_NOTES_1a.csv",
                                                     help="Path to the input model"),
):
    ################### 1 Get Embeddings with BERT ############################

    data_df = pd.read_csv(input_data_file)
    model = SentenceTransformer(input_model)

    all_query_embeddings = []
    for text in data_df["TEXT"]:
        query_embedding = model.encode(text)
        all_query_embeddings.append(query_embedding)
        a=1
    ######################### 2 Similarity in Vectors ################

    #print("Similarity:", util.dot_score(query_embedding, passage_embedding))

    a = 1


if __name__ == "__main__":
    typer.run(main)

a = 1
