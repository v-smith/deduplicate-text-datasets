import re
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from dedup.utils import collate_repeats


def main(
        input_data_file: Path = typer.Option(default="/data/SUBJECT_ID_to_NOTES_1a.csv",
                                             help="Path to the input model"),
        input_repeat_file: Path = typer.Option(default="/tmp/SUBJECT_ID_to_NOTES_1a_7000.train.remove.byterange",
                                               help="Path to the jsonl file of the set"),
        input_pseudo_file: Path = typer.Option(default="/data/SUBJECT_ID_to_NOTES_1b_7000.csv",
                                               help="Path to the input model"),
        input_psuedo_repeat_file: Path = typer.Option(
            default="/tmp/SUBJECT_ID_to_NOTES_1b_7000.train.remove.byterange",
            help="Path to the jsonl file of the set"),
        inspect_dataframes: bool = typer.Option(default=False),
):
    # load data and get data stats
    data_df = pd.read_csv(input_data_file)
    data_df = data_df[data_df.columns[1:]]
    print(len(data_df.index))

    # check for repeats in data
    print(f"Number of Repeated Sequences: {data_df.duplicated(keep=False).value_counts()}")
    # remove repeats
    data_df.drop_duplicates(keep="first", inplace=True)
    print(len(data_df.index))

    # how many notes per patient?
    grouped = data_df.groupby("SUBJECT_ID").count().sort_values(by="TEXT", ascending=False)
    print(f"Total number of patients: {len(grouped)}")  # 200 patients in 7000 records

    # load data file
    data = open(input_data_file, "rb").read()
    # load repeat file
    repeat_dict = collate_repeats(data=data, repeats_file=input_repeat_file)
    # inspect data
    if inspect_dataframes:
        psuedo_data = open(input_pseudo_file, "rb").read()
        psuedo_repeat_dict = collate_repeats(data=psuedo_data, repeats_file=input_psuedo_repeat_file)
        psuedo_repeat_df = pd.DataFrame.from_records(psuedo_repeat_dict)
        repeat_df = pd.DataFrame.from_records(repeat_dict)
        # check if repeat dfs same for both files
        merged_df = pd.merge(repeat_df, psuedo_repeat_df, on="string")
        merged_df.columns = ["n_reps", "string", "n_reps_pseudo"]
        merged_df = merged_df[["n_reps", "n_reps_pseudo", "string"]]

    # how many "repeats" come from duplicate records


    # how many "repeats" come from same patient
    repeats_per_patient_dict = []
    for repeat in tqdm(repeat_dict):
        patients = []
        counter = 0
        for index, row in data_df.iterrows():
            try:
                if repeat["string"] in row["TEXT"]:
                    patients.append(row["SUBJECT_ID"])
            except:
                print(repeat["string"])
                print("\n")
                print(row["TEXT"])
                counter+=1
        if counter > 0:
            print(f"Total Exceptions = {counter}")
            # assert len(patients) == repeat["n_reps"]
        if patients:
            unique_patients = list(set(patients))
            repeats_per_patient_dict.append({"n_reps": repeat["n_reps"], "n_patients": len(unique_patients),
                                             "patient_ids": unique_patients, "string": repeat["string"]})

    #how many sequences are from more than 1 patient
    repeats_from_more_than_one_patient = [x for x in repeats_per_patient_dict if x["n_patients"] > 1]
    print(f"Repeat from more than 1 patient: {len(repeats_from_more_than_one_patient)}")
    repeats_for_patient_109 = [x for x in repeats_per_patient_dict if 109 in x["patient_ids"]]
    print(f"Repeat from patient 109: {len(repeats_for_patient_109)}")
    #repeats_per_p_df = pd.DataFrame.from_records(repeats_per_patient_dict)
    #total_repeats = sum([y["n_reps"] for y in repeats_per_patient_dict])
    a = 1


if __name__ == "__main__":
    typer.run(main)
