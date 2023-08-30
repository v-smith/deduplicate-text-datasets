import numpy as np
import os
import shutil
import json
from transformers import GPT2Tokenizer
import multiprocessing as mp
from collections import defaultdict
import tensorflow as tf

import pickle


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(**feature):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'content-length': _bytes_feature(feature['content-length']),
        'content-type': _bytes_feature(feature['content-type']),
        'text': _bytes_feature(feature['text']),
        'timestamp': _bytes_feature(feature['timestamp']),
        'url': _bytes_feature(feature['url']),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

remove = defaultdict(list)


def run(args):
    this_idx, row = args
    new_row = {'text': row,
               'version_id': '',
               'wikidata_id': '',
               'timestamp': '',
               'url': '',
               'content-length': '',
               'content-type': ''}

    if this_idx in remove_ex:
        for start, end in remove_ex[this_idx][::-1]:
            # print(start,end)
            row = row[:start] + row[end:]

        new_row['text'] = row
    return new_row

import argparse

parser = argparse.ArgumentParser(description='Dedup dataset')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--suffixarray_dir', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--remove', type=str)

args = parser.parse_args()

dataset = args.name
where = args.save_dir

remove = []
fin = open(args.remove)
for line in fin:
    if 'out' in line: break
for line in fin:
    remove.append(list(map(int, line.split())))

sizes = np.frombuffer(open(os.path.join(args.suffixarray_dir, args.name + "." + args.split + ".size"), "rb").read(),
                      dtype=np.uint64)

remove_ex = defaultdict(list)
ptr = 0
for i, byte_start in enumerate(sizes[:-1]):
    byte_end = sizes[i + 1]
    # print(byte_start, byte_end, remove[ptr])
    while ptr < len(remove) and byte_start <= remove[ptr][0] < byte_end:
        # print(remove[ptr])
        assert remove[ptr][1] < byte_end + 6
        # The magic value 6 here corresponds to the 4-byte index prefix followed by \xff\xff.
        remove_ex[i].append((max(int(remove[ptr][0] - byte_start - 6), 0),
                             min(int(remove[ptr][1] - byte_start), byte_end - byte_start)))
        ptr += 1


    root = os.path.join(where + "_dedup", "my_dataset", "1.0.0")
    for f in os.listdir(root):
        if "my_dataset" in f:
            shutil.move(os.path.join(root, f),
                        os.path.join(en, f.replace("my_dataset", dataset)))
        elif f == 'dataset_info.json' and os.path.exists(os.path.join(en, f)):
            json_orig = json.loads(open(os.path.join(en, f)).read())
            json_new = json.loads(open(os.path.join(root, f)).read())
            json_orig['splits'].extend(json_new['splits'])
            open(os.path.join(en, f), "w").write(json.dumps(json_orig))
        else:
            shutil.move(os.path.join(root, f),
                        os.path.join(en, f))
else:
    raise

try:
    os.unlink(os.path.join(where + "_dedup", "my_dataset", "1.0.0", "dataset_info.json"))
except:
    pass
os.rmdir(os.path.join(where + "_dedup", "my_dataset", "1.0.0"))
os.rmdir(os.path.join(where + "_dedup", "my_dataset"))
