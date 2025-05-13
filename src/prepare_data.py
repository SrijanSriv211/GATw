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
parser.add_argument("-s", help="max toks in each data shard", type=int, default=250_000_000)
args = parser.parse_args()

CONFIG = {
	"dataset_path": args.i,
	"outpath": args.o,
	"enc_path": args.e,
	"data_division": args.d,
	"toks_per_shard": args.s
}

"""
Create train-val folders
"""
if not os.path.isdir(os.path.join(CONFIG["outpath"], "train")):
	os.mkdir(os.path.join(CONFIG["outpath"], "train"))

if 0 < CONFIG["data_division"] < 1 and not os.path.isdir(os.path.join(CONFIG["outpath"], "val")):
	os.mkdir(os.path.join(CONFIG["outpath"], "val"))

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
no_val = False if 0 < CONFIG["data_division"] < 1 else True

for file in dataset_files:
	with open(file, "r", encoding="utf-8") as f:
		train_data = json.load(f)

	# get total number chars and total number of unique chars
	unique_chars = len(set().union(*set().union(*train_data)))
	num_chars = sum([len(i) for i in train_data])

	# verbose
	total_chars += num_chars
	total_unique_chars += unique_chars
	print(f"{Fore.YELLOW}{Style.BRIGHT}{file}")
	print(f"{(num_chars/1e6)}M total chars,", f"{unique_chars} unique chars")

	# split val data based on `data_division`
	if not no_val:
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

		# delete useless vars
		del val_size, idx, size

	# verbose
	train_chars = sum([len(i) for i in train_data])
	total_train_chars += train_chars

	if not no_val:
		val_chars = sum([len(i) for i in val_data])
		total_val_chars += val_chars

	print(f"{(train_chars/1e6)}M train chars" + f", {(val_chars/1e6)}M val chars" if not no_val else "")

	# encode data
	train_desc = f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}train chars{Style.RESET_ALL}"
	train_data = [enc.encode(data, allowed_special="all") for data in track(train_data, train_desc)]

	if not no_val:
		val_desc = f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}val chars{Style.RESET_ALL}"
		val_data = [enc.encode(data, allowed_special="all") for data in track(val_data, val_desc)]

	# verbose
	train_tokens = sum([len(i) for i in train_data])
	total_train_tokens += train_tokens

	if not no_val:
		val_tokens = sum([len(i) for i in val_data])
		total_val_tokens += val_tokens

	print(f"{(train_tokens/1e6)}M train tokens" + f", {(val_tokens/1e6)}M val tokens" if not no_val else "")

	# save
	for encoded_data in [(train_data, "train"), (val_data, "val") if not no_val else None]:
		if not encoded_data:
			break

		a, b, c = 0, 1, []
		for i, x in enumerate(encoded_data[0]):
			c.append(x)
			a += len(x)

			if a >= CONFIG["toks_per_shard"] or i >= len(encoded_data[0])-1:
				outpath = f"{CONFIG["outpath"]}/{encoded_data[1]}/{Path(file).stem}_{b}.bin"
				print(outpath)

				pad = len(max(c, key=len))
				numpy.array([i + [0]*(pad-len(i)) for i in c]).tofile(outpath)

				b += 1
				a, c = 0, []
	print()

print(f"{(total_chars/1e6)}M total chars,", f"{total_unique_chars} total unique chars")
print(f"{(total_train_chars/1e6)}M total train chars" + f", {(total_val_chars/1e6)}M total val chars" if not no_val else "")
print(f"{(total_train_tokens/1e6)}M total train tokens" + f", {(total_val_tokens/1e6)}M total val tokens" if not no_val else "")
