from termcolor import colored, cprint
from tqdm import tqdm


def view_all_repeats_terminal(inp_text, match_indices, subject):
    COLOR_MAP = {1: "red", 2: "blue", 3: "green", 4: "cyan", 5: "magenta",
                 6: "cyan", 7: "magenta", 8: "white", 9: "green"}
    count = 0
    coloured_text = ""
    end_previous = 0
    for match in match_indices:
        count += 1
        if count <= 9:
            c = COLOR_MAP[count]
        else:
            c = COLOR_MAP[9]
        coloured_text += inp_text[end_previous:match["start"]]
        coloured_text += colored(inp_text[match["start"]:match["end"]], c, attrs=['reverse', 'bold'])
        end_previous = match["end"]
    coloured_text += inp_text[end_previous:]

    cprint(f"SUBJECT: {subject}", "black", "on_green")
    print("\n")
    print(coloured_text)
    print("\n")
    print("\n")

    return coloured_text


def collate_repeats(data, repeats_file):
    """collate information on repeats"""
    # get repeat indexes
    repeats = open(repeats_file, "r")
    lines = repeats.readlines()
    out_idx = []
    counter = 0
    for line in lines:
        counter += 1
        if "out" in line:
            out_idx.append(counter)
            break
    if out_idx:
        assert len(out_idx) == 1
        lines = lines[out_idx[0]:]

    count = 0
    rep_strings = []
    all_n_reps = []
    rep_dictionary = []
    print("--------Finding repeat sequences----------------")
    for line in tqdm(lines):
        count += 1
        #new_line = line.decode()
        new_line = line.strip().split()
        first_idx = int(new_line[0])
        second_idx = int(new_line[1])
        rep_string = data[first_idx:second_idx]
        rep_strings.append(rep_string.decode())
        n_reps = data.count(data[first_idx:second_idx])
        all_n_reps.append(n_reps)
        rep_dictionary.append({"n_reps": n_reps, "string": rep_string.decode()})
        a = 1
    # checks
    print(len(rep_strings))
    print(len(all_n_reps))
    print(count)
    sorted_reps = sorted(rep_dictionary, key=lambda d: d['n_reps'], reverse=True)
    return sorted_reps
