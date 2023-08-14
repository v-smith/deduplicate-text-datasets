import re
from pathlib import Path

import pandas as pd
import typer

from dedup.utils import collate_repeats, view_all_repeats_terminal


def main(
        input_data_file: Path = typer.Option(default="data/SUBJECT_ID_to_NOTES_1a_7000.csv",
                                             help="Path to the input model"),
        input_repeat_file: Path = typer.Option(default="tmp/SUBJECT_ID_to_NOTES_1a_7000.train.remove.byterange",
                                               help="Path to the jsonl file of the set")
):
    # open data file
    data = open(input_data_file, "rb").read()

    # get repeats
    repeat_dict = collate_repeats(data=data, repeats_file=input_repeat_file)

    # see duplicate phrases in context
    data_df = pd.read_csv(input_data_file)
    for repeat in repeat_dict[:1]:
        repeat_string = repeat["string"]
        print(f"Repeated String: {repeat_string}")
        for subject, text in zip(data_df["SUBJECT_ID"], data_df["TEXT"]):
            match_indices = []
            for match in re.finditer(repeat_string, text):
                # print("Match at index % s, % s" % (match.start(), match.end()))
                match_indices.append(dict(start=match.start(), end=match.end()))
            if match_indices:
                coloured_text = view_all_repeats_terminal(inp_text=text, match_indices=match_indices, subject=subject)


if __name__ == "__main__":
    typer.run(main)
