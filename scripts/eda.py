import pandas as pd

#open data file
data = open("../data/SUBJECT_ID_to_NOTES_1a_7000.csv", "rb").read()

#get repeat indexes
repeats = open("../tmp/SUBJECT_ID_to_NOTES_1a_ready.train.remove.byterange", "rb")
lines = repeats.readlines()

#collate information on repeats
count = 0
rep_strings = []
all_n_reps = []
rep_dictionary = []
for line in lines:
    count +=1
    new_line = line.decode()
    new_line = new_line.strip().split()
    first_idx = int(new_line[0])
    second_idx = int(new_line[1])
    rep_string = data[first_idx:second_idx]
    rep_strings.append(rep_string.decode())
    n_reps = data.count(data[first_idx:second_idx])
    all_n_reps.append(n_reps)
    rep_dictionary.append({"n_reps": n_reps, "string": rep_string.decode()})
    a=1

#checks
print(len(rep_strings))
print(len(all_n_reps))
print(count)

#analyse repeats
sorted_reps = sorted(rep_dictionary, key=lambda d: d['n_reps'], reverse=True)
df = pd.DataFrame.from_records(sorted_reps)
a=1



