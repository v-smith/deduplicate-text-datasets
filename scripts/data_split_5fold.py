from pathlib import Path

import numpy as np
import pandas as pd
import typer


def main(
        input_data_file_path: Path = typer.Option(default="../mimic/files/clinical-bert-mimic-notes/setup_outputs/SUBJECT_ID_to_NOTES_1a.csv",
                                                  help="Path to the input model"),
        output_data_file_name: Path = typer.Option(default="../mimic/files/clinical-bert-mimic-notes/setup_outputs/split/10fold/SUBJECT_ID_to_NOTES_1a",
                                                   help="Path to the input model"),
        splits: int = typer.Option(default=10, help="Number of Splits of Data")
):
    df = pd.read_csv(input_data_file_path, index_col=False)
    print(len(df.index)/splits)
    shuffled = df.sample(frac=1)
    result = np.array_split(shuffled, indices_or_sections=splits)

    counter = 0
    for part in result:
        counter += 1
        print(f"Len of split {counter} is: {len(part)}")
        part.to_csv(path_or_buf=f"{output_data_file_name}_split{counter}.csv")


if __name__ == "__main__":
    typer.run(main)
