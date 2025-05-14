from encoder.bytepair import Encoder
from rich.progress import track
from colorama import Style, Fore, init
from pathlib import Path
import argparse, numpy, json, os

init(autoreset=True)

parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
parser.add_argument("-i", help="dataset path", required=True)
parser.add_argument("-o", help="output path", required=True)
parser.add_argument("-e", help="encoder path", required=True)
parser.add_argument("-d", help="train-val data division ratio", type=float, default=0.9)
parser.add_argument("-s", help="max chars in each data shard", type=int, default=2_000_000_000)
args = parser.parse_args()

CONFIG = {
	"dataset_path": args.i,
	"outpath": args.o,
	"enc_path": args.e,
	"data_division": args.d,
	"chars_per_shard": args.s
}

"""
Load encoder
"""
enc = Encoder()
enc.load(CONFIG["enc_path"])
data = []

"""
Pretraining dataset
"""
dataset_files = [os.path.join(CONFIG["dataset_path"], i) for i in os.listdir(CONFIG["dataset_path"])]
dataset_files = sorted(dataset_files, key=os.path.getsize)

# python src/prepare_data.py -i data/base/json -o data/base -e bin/cl4k.bin
total_unique_chars, total_chars = 0, 0
total_train_chars, total_val_chars = 0, 0
total_train_tokens, total_val_tokens = 0, 0
lensum = lambda x: sum([len(i) for i in x])
save_file_idx = [0, 0]

def save_data(data, split, file):
	# if not os.path.isdir(os.path.join(CONFIG["outpath"], split)):
	# 	os.mkdir(os.path.join(CONFIG["outpath"], split))

	if split == "train": save_file_idx[0] += 1
	else: save_file_idx[1] += 1

	idx = save_file_idx[0] if split == "train" else save_file_idx[1]
	outpath = f"{CONFIG["outpath"]}/{split}/{Path(file).stem}" + (f"_{idx}" if idx > 0 else "") + ".bin"
	print(outpath)

	print(lensum(data))
	pad = len(max(data, key=len))
	data = numpy.array([i + [-1]*(pad-len(i)) for i in data])
	print(lensum(data))
	# data.tofile(outpath)

def encode_data(data, split, file):
	global save_file_idx
	num_chars = lensum(data)

	for i, x in enumerate(track(data, f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}{split} chars{Style.RESET_ALL}")):
		data[i] = enc.encode(x, allowed_special="all")

	num_tokens = lensum(data)
	print(f"{(num_chars/1e6)}M {split} chars,", f"{(num_tokens/1e6)}M {split} tokens")

	# save
	save_data(data, split, file)
	return num_chars, num_tokens

# split train and val data based on `data_division`
def split_data(train_data):
	val_size = int(num_chars * (1 - CONFIG["data_division"]))
	val_data, idx, size = [], [], 0
	for i, x in enumerate(train_data):
		if size >= val_size:
			break

		elif i % (CONFIG["data_division"] * 10) != 0:
			continue

		val_data.append(x)
		idx.append(i)
		size += len(x)

	# remove items from `train_data` which are now a part of `val_data`
	idx.sort(reverse=True)
	[train_data.pop(i) for i in idx]
	return val_data

for file in dataset_files:
	print(f"{Fore.YELLOW}{Style.BRIGHT}{file}")
	with open(file, "r", encoding="utf-8") as f:
		train_data = json.load(f)

	# get total number chars and total number of unique chars
	unique_chars = len(set().union(*set().union(*train_data)))
	num_chars = lensum(train_data)
	save_file_idx = [0, 0]

	# verbose
	total_chars += num_chars
	total_unique_chars += unique_chars
	print(f"{(num_chars/1e6)}M total chars,", f"{unique_chars} unique chars")

	# load data in chunks
	size = 0
	chunk_ranges = [0]
	for i, x in enumerate(train_data):
		size += len(x)
		if size > CONFIG["chars_per_shard"] or i+1 >= len(train_data):
			chunk_ranges.append(i+1)
			size = 0
	chunk_ranges = list(zip(chunk_ranges, chunk_ranges[1:]))
	del train_data

	# now prepare data
	for i in chunk_ranges:
		with open(file, "r", encoding="utf-8") as f:
			train_data = json.load(f)[i[0]:i[1]]

		# split and save
		val_data = split_data(train_data)
		num_train_chars, num_train_tokens = encode_data(train_data, "train", file)
		num_val_chars, num_val_tokens = encode_data(val_data, "val", file)
		print()

		total_train_chars += num_train_chars
		total_train_tokens += num_train_tokens
		total_val_chars += num_val_chars
		total_val_tokens += num_val_tokens
		del train_data, val_data

print(f"{(total_chars/1e6)}M total chars,", f"{total_unique_chars} total unique chars")
print(f"{(total_train_chars/1e6)}M total train chars" + f", {(total_val_chars/1e6)}M total val chars")
print(f"{(total_train_tokens/1e6)}M total train tokens" + f", {(total_val_tokens/1e6)}M total val tokens")
