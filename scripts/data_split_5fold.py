from pathlib import Path

import numpy as np
import pandas as pd
import typer


def main(
        input_data_file_path: Path = typer.Option(default="/datadrive/mimic/files/clinical-bert-mimic-notes/setup_outputs/SUBJECT_ID_to_NOTES_1b.csv",
                                                  help="Path to the input model"),
        output_data_file_name: Path = typer.Option(default="/datadrive/mimic/files/clinical-bert-mimic-notes/setup_outputs/split/SUBJECT_ID_to_NOTES_1b",
                                                   help="Path to the input model"),
):
    df = pd.read_csv(input_data_file_path, index_col=False)
    shuffled = df.sample(frac=1)
    result = np.array_split(shuffled, 5)

    counter = 0
    for part in result:
        counter += 1
        part.to_csv(path_or_buf=f"{output_data_file_name}_split{counter}.csv")


if __name__ == "__main__":
    typer.run(main)
