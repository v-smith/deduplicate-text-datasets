from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from dedup.utils import collate_repeats


def main(
        input_data_file: Path = typer.Option(default="../data/SUBJECT_ID_to_NOTES_1a_10percent.csv",
                                             help="Path to the input model"),
        input_repeat_file: Path = typer.Option(default="../tmp/SUBJECT_ID_to_NOTES_1a_10percent.train.remove.byterange",
                                               help="Path to the jsonl file of the set"),
        inspect_dataframes: bool = typer.Option(default=False),
        input_pseudo_file: Path = typer.Option(default="data/SUBJECT_ID_to_NOTES_1b_7000.csv",
                                               help="Path to the input model"),
        input_psuedo_repeat_file: Path = typer.Option(
            default="tmp/SUBJECT_ID_to_NOTES_1b.train.remove.byterange",
            help="Path to the jsonl file of the set"),
):
    # load data file
    data = open(input_data_file, "rb").read()
    print("------Data File Open--------")
    # load repeat file
    repeat_dict = collate_repeats(data=data, repeats_file=input_repeat_file)
    #repeat_df = pd.DataFrame(repeat_dict)
    print("---------Repeats Collated-----------")
    # inspect data
    if inspect_dataframes:
        #psuedo_data = open(input_pseudo_file, "rb").read()
        #psuedo_repeat_dict = collate_repeats(data=psuedo_data, repeats_file=input_psuedo_repeat_file)
        #psuedo_repeat_df = pd.DataFrame.from_records(psuedo_repeat_dict)
        repeat_df = pd.DataFrame.from_records(repeat_dict)
        # check if repeat dfs same for both files
        #merged_df = pd.merge(repeat_df, psuedo_repeat_df, on="string")
        #merged_df.columns = ["n_reps", "string", "n_reps_pseudo"]
        #merged_df = merged_df[["n_reps", "n_reps_pseudo", "string"]]

    # load data and get data stats
    data_df = pd.read_csv(input_data_file, index_col=False)
    data_df = data_df[data_df.columns[1:]]
    print(f"Columns = {data_df.columns}")
    print(f"Total Len of data {len(data_df.index)}")

    # check for repeats in data
    print(f"Number of Repeated Rows: {data_df.duplicated(keep=False).value_counts()}")
    # remove repeats
    data_df.drop_duplicates(keep="first", inplace=True)
    print(f"Total Len of data without repeats: {len(data_df.index)}")

    # how many notes per patient?
    grouped = data_df.groupby("SUBJECT_ID").count().sort_values(by="TEXT", ascending=False)
    print(f"Total number of patients: {len(grouped)}")  # 200 patients in 7000 records

    # how many "repeats" come from duplicate records

    # how many "repeats" come from same patient
    repeats_per_patient_dict = []
    print("---------Finding Repeats Per Patient------------")
    data_series = data_df.set_index(keys="SUBJECT_ID", drop=True).squeeze()
    for repeat in tqdm(repeat_dict):
        patients = []
        counter = 0
        for i, row in data_series.items():
            try:
                if repeat["string"] in row:
                    patients.append(i)
            except:
                print(repeat["string"])
                print("\n")
                print(row)
                counter+=1
        if counter > 0:
            print(f"Total Exceptions = {counter}")
        if patients:
            unique_patients = list(set(patients))
            repeats_per_patient_dict.append({"n_reps": repeat["n_reps"], "n_patients": len(unique_patients),
                                             "patient_ids": unique_patients, "string": repeat["string"]})

    #repeats_per_patient_df = pd.DataFrame(repeats_per_patient_dict)

    #how many sequences are from more than 1 patient
    repeats_from_more_than_one_patient = [x for x in repeats_per_patient_dict if x["n_patients"] > 1]
    print(f"Repeat from more than 1 patient: {len(repeats_from_more_than_one_patient)}")
    repeats_for_patient_109 = [x for x in repeats_per_patient_dict if 109 in x["patient_ids"]]
    print(f"Repeat from patient 109: {len(repeats_for_patient_109)}")
    #repeats_per_p_df = pd.DataFrame.from_records(repeats_per_patient_dict)
    total_repeats = sum([y["n_reps"] for y in repeats_per_patient_dict])
    print(f"Total Repeats: {total_repeats}")
    a = 1


if __name__ == "__main__":
    typer.run(main)
