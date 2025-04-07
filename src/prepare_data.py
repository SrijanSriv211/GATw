from encoder.bytepair import Encoder
import torch, pickle, os

def save_distributed_data(path, name, data, distribution):
    distributed_data = [] # (path, data)

    # distribute the data
    if distribution:
        if not os.path.isdir(f"{path}\\{name}"):
            os.mkdir(f"{path}\\{name}")

        count = 1
        for i in range(0, len(data), int(len(data) / distribution)):
            d = data[i:i+distribution]
            print(f"{len(d)}: [{i}:{i+distribution}]", f"{path}\\{name}\\{count}.bin")
            distributed_data.append((f"{path}\\{name}\\{count}.bin", d))
            count += 1

    else:
        distributed_data.append((f"{path}\\{name}.bin", data))
    del data

    # save the data
    for p, d in distributed_data:
        with open(p, "wb") as f:
            pickle.dump(d, f)

def prepare_data(encoded_data, path="data", data_division=1, convert_to_tensor=True, distribution=None):
    data = torch.tensor(encoded_data, dtype=torch.long) if convert_to_tensor else encoded_data

    # print the number of tokens
    print(f"{(len(data)/1e6)}M", "total tokens")

    if 0 < data_division < 1:
        # train and test splits
        n = int(data_division * len(data)) # the first (data_division * 100)% will be train, rest val
        train_data = data[:n]
        val_data = data[n:] if 0 < data_division < 1 else data[:n]
        del data # free up some memory

        print(f"{(len(train_data)/1e6)}M", "train tokens,", f"{(len(val_data)/1e6)}M", "test tokens")

        # save the data
        save_distributed_data(path, "val", val_data, distribution)
        del val_data # again free up some memory

    save_distributed_data(path, "train", train_data if 0 < data_division < 1 else data, distribution)

enc = Encoder()
enc.load("bin\\cl5k.bin")
data = []

"""
Pretraining dataset
"""
path = "data\\base"
files = os.listdir(path)

for i in files:
	with open(f"{path}\\{i}", "r", encoding="utf-8") as f:
		data.append(f"========== {i} ==========\n{f.read()}")

"""
Save dataset
"""
data = "\n\n".join(data)

# with open("data\\data.txt", "w", encoding="utf-8") as f:
# 	f.write(data + "\n")

print(f"{(len(data)/1e6)}M total chars", f"{(len(set(data)))} unique chars")
prepare_data(enc.encode(data, allowed_special="all"), "bin", 0.8, distribution=None)
