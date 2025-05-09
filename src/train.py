from shared.utils import calc_total_time, kprint
from models.gpt import GPTConfig, GPT, sample
from colorama import Style, Fore, init
from encoder.bytepair import Encoder
from contextlib import nullcontext
from rich.progress import track
from pathlib import Path
import warnings, pickle, pandas, random, time, math, os
import torch._inductor.config as config
import torch.amp, torch, json, sys

# supress pytorch's future warning:
# You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.
# It is possible to construct malicious pickle data which will execute arbitrary code during unpickling
# (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
# In a future release, the default value for `weights_only` will be flipped to `True`.
# This limits the functions that could be executed during unpickling.
# Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
# We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file.
# Please open an issue on GitHub for any issues related to this experimental feature.
warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset=True)

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "scripts/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	CONFIG = json.load(f)
SAVE_PATH = Path(CONFIG["save_path"])
TXT_SAVE_PATH = list(SAVE_PATH.parts)
TXT_SAVE_PATH[-1] = SAVE_PATH.stem
TXT_SAVE_PATH = "/".join(TXT_SAVE_PATH) + ".txt"

if CONFIG["init_from"] == "scratch":
	with open(TXT_SAVE_PATH, "w", encoding="utf-8") as f:
		f.write("")

kprint(f"```config.json\n{json.dumps(CONFIG, indent=4)}\n```", filename=TXT_SAVE_PATH, println=False)

# set device
device = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"] == "auto" else CONFIG["device"]

# init seed
torch.manual_seed(CONFIG["seed"]) if CONFIG["seed"] != "auto" else None
random.seed(CONFIG["seed"]) if CONFIG["seed"] != "auto" else None

# "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

# print the device
kprint("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}", f"{Fore.WHITE}{Style.BRIGHT}({torch.initial_seed()})", filename=TXT_SAVE_PATH)

def from_scratch():
	hyperparams = dict(dropout=CONFIG["dropout"])
	# read off the created CONFIG params, so we can store them into checkpoint correctly
	for k in ["n_layer", "n_head", "n_embd", "n_hidden", "use_rope", "rope_base", "block_size", "vocab_size", "beta1", "beta2"]:
		hyperparams[k] = CONFIG[k]
	# automatically set `n_hidden` for feedforward network if not set already
	if any([hyperparams["n_hidden"] == i for i in ["4x_embd", "auto", None]]):
		hyperparams["n_hidden"] = hyperparams["n_embd"] * 4

	gptconf = GPTConfig(**hyperparams)
	# create an instance of GPT
	model = GPT(gptconf)
	model.to(device)

	# optimizer
	optimizer = model.configure_optimizers(CONFIG["weight_decay"], CONFIG["learning_rate"], CONFIG["device"])

	# a dict for keep track of all the losses to be plotted.
	metrics = {
		"train": [],
		"eval": [],
		"val": [],
		"mfu": [],
		"lr": []
	}
	iter_num = 0
	best_loss = 0

	return model, optimizer, hyperparams, iter_num, best_loss, metrics

def from_pretrained(checkpoint):
	metrics = checkpoint["metrics"] if "metrics" in checkpoint.keys() else {
		"train": [],
		"eval": [],
		"val": [],
		"mfu": [],
		"lr": []
	}

	# load the state dict and current iteration number of the model
	iter_num = checkpoint["iter_num"] + 1
	best_loss = checkpoint["best_loss"] if "best_loss" in checkpoint.keys() else 0

	hyperparams = dict(dropout=CONFIG["dropout"])
	# read off the created config params, so we can store them into checkpoint correctly
	for k in ["n_layer", "n_head", "n_embd", "n_hidden", "use_rope", "rope_base", "block_size", "vocab_size", "beta1", "beta2"]:
		hyperparams[k] = checkpoint["hyperparams"][k]
	# automatically set `n_hidden` for feedforward network if not set already
	if any([hyperparams["n_hidden"] == i for i in ["4x_embd", "auto", None]]):
		hyperparams["n_hidden"] = hyperparams["n_embd"] * 4

	gptconf = GPTConfig(**hyperparams)

	# create an instance of GPT
	model = GPT(gptconf)

	# remove `_orig_mod.` prefix from state_dict (if it's there)
	state_dict = checkpoint["model"]
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	model.load_state_dict(state_dict)
	model.to(device)

	# optimizer
	optimizer = model.configure_optimizers(CONFIG["weight_decay"], CONFIG["learning_rate"], CONFIG["device"])
	optimizer.load_state_dict(checkpoint["optimizer"])

	# crop down the model block size if desired, using model surgery
	if CONFIG["block_size"] < hyperparams["block_size"]:
		model.crop_block_size(CONFIG["block_size"])
		hyperparams["block_size"] = CONFIG["block_size"] # so that the checkpoint will have the right value

	return model, optimizer, hyperparams, iter_num, best_loss, metrics

# init model and optimizer
if CONFIG["init_from"] == "scratch":
	model, optimizer, hyperparams, iter_num, best_loss, metrics = from_scratch()

elif CONFIG["init_from"].startswith("pretrained,"):
	model, optimizer, hyperparams, iter_num, best_loss, metrics = from_pretrained(torch.load(CONFIG["init_from"][11:]))

# load all the files
train_data, val_data = None, None
train_data_len, val_data_len = 0, 0
if CONFIG["load_from_file"]:
	train_data = [torch.tensor(i, dtype=torch.long) for i in pandas.read_parquet(CONFIG["train_data"], engine="pyarrow")["tok"].tolist()]
	val_data = [torch.tensor(i, dtype=torch.long) for i in pandas.read_parquet(CONFIG["val_data"], engine="pyarrow")["tok"].tolist()]

	for i in train_data:
		train_data_len += len(i)

	for i in val_data:
		val_data_len += len(i)

data_len = train_data_len + val_data_len

# print the number of tokens
color = f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}" if CONFIG["use_rope"] else f"{Fore.LIGHTRED_EX}{Style.BRIGHT}"
kprint("using RoPE:", f"{color}{CONFIG["use_rope"]}", filename=TXT_SAVE_PATH)
kprint(f"{Fore.WHITE}{Style.BRIGHT}{(data_len/1e6)}M", "total tokens", filename=TXT_SAVE_PATH)
kprint(
	f"{Fore.WHITE}{Style.BRIGHT}{(len(train_data)/1e6)}M", "train entries,",
	f"{Fore.WHITE}{Style.BRIGHT}{(len(val_data)/1e6)}M", "test entries\n"
	f"{Fore.WHITE}{Style.BRIGHT}{(train_data_len/1e6)}M", "train tokens,",
	f"{Fore.WHITE}{Style.BRIGHT}{(val_data_len/1e6)}M", "test tokens",
	f"   {Fore.WHITE}{Style.DIM}(using train tokens as test tokens)" if train_data_len == val_data_len else "",
	filename=TXT_SAVE_PATH
)
del data_len, train_data_len, val_data_len # these are useless vars, delete them

def get_trained_model(model, optimizer):
	return {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"hyperparams": hyperparams,
		"device": device,
		"metrics": metrics,
		"iter_num": iter_num,
		"best_loss": best_loss
	}

def _load_data(path):
	if CONFIG["load_from_file"]:
		return train_data if path == CONFIG["train_data"] else val_data

	else:
		if not CONFIG["load_from_file"]:
			files = os.listdir(path)
			random.shuffle(files)

		with open(f"{path}/{files[0]}" if not CONFIG["load_from_file"] else path, "rb") as f:
			return pickle.load(f)

# data loading
# generate a small batch of data of inputs x and targets y
def get_batch(split):
	# we reload data every batch to avoid a memory leak
	path = CONFIG["train_data"] if split == "train" else CONFIG["val_data"]
	data = _load_data(path)

	# get `batch_size` number of random entries
	ix = torch.randint(len(data), (CONFIG["batch_size"],))
	k = {}

	# get `block_size + 4` length of data for each batch
	for i in ix:
		k[i.item()] = data[i]

		# `CONFIG["block_size"] + 4` to ensure that `k` is always greater than block_size
		c = 1
		while len(k[i.item()]) < CONFIG["block_size"]+4:
			k[i.item()] = torch.cat((k[i.item()], data[0] if i+c >= len(data) else data[i+c]))
			c += 1

	# randomly select starting position
	p = {i: random.randint(0, len(k[i]) - CONFIG["block_size"] - 1) if random.randint(0, 2) == 0 else 0 for i in k}

	# prepare the train and val dataset
	x = torch.stack([k[i][p[i] : p[i] + CONFIG["block_size"]] for i in k])
	y = torch.stack([k[i][p[i] + 1 : p[i] + CONFIG["block_size"] + 1] for i in k])
	x, y = x.to(device), y.to(device)
	return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(eval_iters):
	out = {}
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(eval_iters)
		for k in track(range(eval_iters), description=f"{Fore.WHITE}{Style.BRIGHT}calc {Fore.WHITE}{Style.DIM}{split} loss{Style.RESET_ALL}"):
			X, Y = get_batch(split)
			with ctx:
				logits, loss = model(X, Y)

			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < CONFIG["warmup_iters"]:
        return CONFIG["learning_rate"] * (it + 1) / (CONFIG["warmup_iters"] + 1)

    # 2) if it > lr_decay_iters, return min learning rate
    if it > CONFIG["lr_decay_iters"]:
        return CONFIG["min_lr"]

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - CONFIG["warmup_iters"]) / (CONFIG["lr_decay_iters"] - CONFIG["warmup_iters"])

    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

# https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
class adaptive_early_stopping:
    def __init__(self, base_patience=5, delta=0.01):
        self.base_patience = base_patience
        self.delta = delta
        self.wait_count = 0
        self.best_score = None
        self.dynamic_patience = self.base_patience

    def step(self, val_loss):
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.wait_count = 0
            self.dynamic_patience = self.base_patience # reset to base

        else:
            self.wait_count += 1
            # adjust patience if improvement is near
            if self.wait_count >= (self.base_patience * 0.8):
                self.dynamic_patience += 1

            if self.wait_count >= self.dynamic_patience:
                return True # signal to stop training
        return False

# report number of parameters
kprint(f"{Fore.WHITE}{Style.BRIGHT}{model.get_num_params()/1e6}M", "parameters", filename=TXT_SAVE_PATH)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=False)

if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
# compile the model
if CONFIG["compile"]:
	kprint(f"compiling the model... {Fore.WHITE}{Style.DIM}(takes a ~minute)", filename=TXT_SAVE_PATH)
	#NOTE: backend="inductor" is giving some errors so switched to aot_eager.
	model = torch.compile(model, backend="aot_eager") # requires PyTorch 2.0

# training loop
X, Y = get_batch("train") # fetch the very first batch
start_time = time.time()
eval_t0 = time.time()
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0 if metrics["mfu"] == [] else metrics["mfu"][-1]
training_loop = True
stop_early = adaptive_early_stopping()
training_sample = sample()
enc = Encoder()
enc.load(CONFIG["encoder_path"])

while training_loop:
	try:
		# determine and set the learning rate for this iteration
		lr = get_lr(iter_num) if CONFIG["decay_lr"] else CONFIG["learning_rate"]
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr
		metrics["lr"].append(lr)

		# save checkpoint
		if CONFIG["checkpoints"] != None and iter_num > 0 and iter_num % CONFIG["checkpoints"]["interval"] == 0:
			if not os.path.isdir(CONFIG["checkpoints"]["path"]):
				os.mkdir(CONFIG["checkpoints"]["path"])

			kprint(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{iter_num}", filename=TXT_SAVE_PATH)
			torch.save(get_trained_model(model, optimizer), f"{CONFIG["checkpoints"]["path"]}/s{iter_num}.bin")

		# generate some sample text
		if CONFIG["gen_interval"] != None and iter_num > 0 and iter_num % CONFIG["gen_interval"] == 0:
			training_sample.load(get_trained_model(model, optimizer), True)

			for _ in range(CONFIG["gen_iters"]):
				out = enc.decode(training_sample.generate(None, length=256))
				kprint(f"{Fore.WHITE}{Style.DIM}```s{iter_num}.bin\n{out}\n```\n", filename=TXT_SAVE_PATH)

		# evaluate the loss on train/val sets and write checkpoints
		if iter_num > 0 and iter_num % CONFIG["eval_interval"] == 0:
			losses = estimate_loss(CONFIG["eval_iters"])
			# timing and logging
			eval_t1 = time.time()
			eval_dt = eval_t1 - eval_t0
			eval_t0 = eval_t1

			kprint(
				f"{Fore.WHITE}{Style.BRIGHT}step",
				f"{Fore.WHITE}{Style.DIM}[{iter_num}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.7f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(eval_dt)}",
				filename=TXT_SAVE_PATH
			)

			metrics["train"].append(losses["train"])
			metrics["val"].append(losses["val"])

			if stop_early.step(losses["val"]):
				kprint(f"{Fore.RED}{Style.BRIGHT}early stopping.")
				training_loop = False
				break

		# forward backward update, with optional gradient accumulation to simulate larger batch size
		# and using the GradScaler if data type is float16
		for micro_step in range(CONFIG["gradient_accumulation_steps"]):
			with ctx:
				logits, loss = model(X, Y)
				loss = loss / CONFIG["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation

			# immediately async prefetch next batch while model is doing the forward pass on the GPU
			X, Y = get_batch("train")
			# backward pass, with gradient scaling if training in fp16
			scaler.scale(loss).backward()

		# clip the gradient
		if CONFIG["grad_clip"] != 0.0:
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

		# step the optimizer and scaler if training in fp16
		scaler.step(optimizer)
		scaler.update()

		# flush the gradients as soon as we can, no need for this memory anymore
		optimizer.zero_grad(set_to_none=True)

		# timing and logging
		if iter_num % CONFIG["log_interval"] == 0:
			t1 = time.time()
			dt = t1 - t0
			t0 = t1

			# get loss as float. note: this is a CPU-GPU sync point
			# scale up to undo the division above, approximating the true total loss (exact would have been a sum)
			lossf = loss.item() * CONFIG["gradient_accumulation_steps"]

			if local_iter_num >= 5: # let the training loop settle a bit
				mfu = model.estimate_mfu(CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * CONFIG["log_interval"], dt) # https://github.com/karpathy/nanoGPT/pull/527/files
				running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

			kprint(
				f"{Fore.WHITE}{Style.BRIGHT}iter",
				f"{Fore.WHITE}{Style.DIM}[{iter_num}/{CONFIG["max_iters"]}]"
				f"{Fore.RESET}{Style.RESET_ALL}:",
				f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"mfu {Fore.WHITE}{Style.BRIGHT}{running_mfu*100:.2f}"
				f"{Fore.RESET}{Style.RESET_ALL},",
				f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(dt)}",
				filename=TXT_SAVE_PATH
			)
			metrics["mfu"].append(running_mfu)
			metrics["eval"].append(lossf)

		iter_num += 1
		local_iter_num += 1

		# termination conditions
		if iter_num > CONFIG["max_iters"]:
			break

	except KeyboardInterrupt:
		kprint("type", filename=TXT_SAVE_PATH)
		kprint(f"{Fore.WHITE}{Style.BRIGHT}1. {Fore.WHITE}{Style.DIM}`y` {Style.RESET_ALL}to stop training.", filename=TXT_SAVE_PATH)
		kprint(f"{Fore.WHITE}{Style.BRIGHT}2. {Fore.WHITE}{Style.DIM}`n` {Style.RESET_ALL}to continue training.", filename=TXT_SAVE_PATH)
		kprint(f"{Fore.WHITE}{Style.BRIGHT}3. {Fore.WHITE}{Style.DIM}`s` {Style.RESET_ALL}to save model.", filename=TXT_SAVE_PATH)
		kprint(f"{Fore.WHITE}{Style.BRIGHT}4. {Fore.WHITE}{Style.DIM}`r` {Style.RESET_ALL}to reload config.json.", filename=TXT_SAVE_PATH)

		while True:
			inp = input("> ")

			if inp == "y":
				kprint(f"{Fore.RED}{Style.BRIGHT}early stopping.", filename=TXT_SAVE_PATH)
				training_loop = False
				break

			elif inp == "n":
				kprint(f"{Fore.GREEN}{Style.BRIGHT}continue training.", filename=TXT_SAVE_PATH)
				break

			elif inp == "s":
				kprint(f"{Fore.YELLOW}{Style.BRIGHT}saving model.")
				kprint("total time:", calc_total_time(time.time() - start_time))
				torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])

			elif inp == "r":
				kprint(f"{Fore.YELLOW}{Style.BRIGHT}config.json{Style.RESET_ALL} reloaded.", filename=TXT_SAVE_PATH)
				with open(CONFIG_PATH, "r", encoding="utf-8") as f:
					CONFIG = json.load(f)

			else:
				kprint(f"{Fore.RED}{Style.DIM}Wrong option.")

kprint("total time:", calc_total_time(time.time() - start_time), filename=TXT_SAVE_PATH)
torch.save(get_trained_model(model, optimizer), CONFIG["save_path"])
