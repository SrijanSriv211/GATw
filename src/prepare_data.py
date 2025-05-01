from encoder.bytepair import Encoder
from rich.progress import track
from colorama import Style, Fore, init
import pandas, json, sys, os

init(autoreset=True)

def prepare_data(encoded_data, path="data", data_division=1):
    # print the number of tokens
	total_tokens = 0
	for i in encoded_data:
		total_tokens += len(i)
	print(f"{(total_tokens/1e6)}M", "total tokens")

	# train and test splits
	train_data, val_data = [], []
	for i, x in enumerate(encoded_data):
		if 0 < data_division < 1 and i % (data_division * 10) == 0:
			val_data.append(x)

		else:
			train_data.append(x)
	del encoded_data

    # print the number of tokens
	train_tokens, val_tokens = 0, 0
	for i in train_data:
		train_tokens += len(i)

	for i in val_data:
		val_tokens += len(i)

	print(
		f"{(len(train_data)/1e6)}M train entries,", f"{(train_tokens/1e6)}M train tokens,",
		f"{(len(val_data)/1e6)}M test entries,", f"{(val_tokens/1e6)}M test tokens"
	)

	# save data
	pandas.DataFrame({"tok": train_data}).to_parquet(f"{path}\\train.parquet", engine="pyarrow")
	pandas.DataFrame({"tok": val_data}).to_parquet(f"{path}\\val.parquet", engine="pyarrow") if val_tokens > 0 else None

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "scripts\\prep_data_config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	CONFIG = json.load(f)

enc = Encoder()
enc.load(CONFIG["enc_path"])
data = []

"""
Pretraining dataset
"""
path = CONFIG["dataset_path"]
files = os.listdir(path)

total_chars = 0
unique_chars = set()
for i in files:
	with open(f"{path}\\{i}", "r", encoding="utf-8") as f:
		for k in track(json.load(f), f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}{i}{Style.RESET_ALL}"):
			data.append(enc.encode(k, allowed_special="all"))
			total_chars += len(k)
			unique_chars.update(set(k))
unique_chars = len(unique_chars)

print(f"{(len(data)/1e6)}M total entries,", f"{(total_chars/1e6)}M total chars,", f"{unique_chars} unique chars")
prepare_data(data, CONFIG["outpath"], data_division=CONFIG["data_division"])
