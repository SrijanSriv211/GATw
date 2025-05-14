# GATw
### Generative Alphabet Transformers write

## How to?
### Use encoder
Pretrain encoder on some text file

```shell
$ python src/train_enc.py -h
usage: train_enc.py [-h] -i I -o O -v V [-r R] [-s S]

A powerful text encryption and decryption program.

options:
  -h, --help  show this help message and exit
  -i I        dataset path
  -o O        output path
  -v V        vocab size
  -r R        text range
  -s S        special tokens

$ python src/train_enc.py -i data/base/data.txt -o bin/cl4k.bin -v 4096
```

Use the pretrained encoder

```python
from encoder.bytepair import Encoder

enc = Encoder()
enc.load("bin/cl4k.bin")

enctxt = enc.encode("Hello world!", allowed_special="all")
out = enc.decode(enctext)
```

### Prepare train and test dataset
> [!NOTE]
> Dataset path must contain `.json` files with the structure `["Hello world!", "This is another sentence", "An other sentence"]`

```shell
$ python src/prepare_data.py -h
usage: prepare_data.py [-h] -i I -o O -e E [-d D] [-s S] [-c C]

A powerful text encryption and decryption program.

options:
  -h, --help  show this help message and exit
  -i I        dataset path
  -o O        output path
  -e E        encoder path
  -d D        train-val data division ratio
  -s S        max toks in each data shard
  -c C        context length
```

### Train the model
1. Create `scripts\config.json`:

```json
{
	"load_from_file": true,
	"train_data": "bin/train.bin",
	"val_data": "bin/val.bin",
	"init_from": "scratch",

	"checkpoints": {
		"path": "bin/checkpoints",
		"interval": 1000
	},
	"save_path": "bin/GATw.bin",

	"max_iters": 100000,
	"eval_interval": 1000,
	"log_interval": 100,
	"eval_iters": 100,
	"encoder_path": "bin/cl4k.bin",
	"gen_interval": 500,
	"gen_iters": 3,

	"gradient_accumulation_steps": 4,
	"batch_size": 16,
	"block_size": 256,

	"vocab_size": 4096,
	"n_layer": 4,
	"n_head": 4,
	"n_embd": 256,
	"n_hidden": "4x_embd",
	"dropout": 0,

	"learning_rate": 5e-5,
	"weight_decay": 0.1,
	"grad_clip": 1,

	"decay_lr": true,
	"warmup_iters": 1000,
	"lr_decay_iters": 100000,
	"min_lr": 5e-6,
	"beta1": 0.9,
	"beta2": 0.95,

	"device": "cpu",
	"seed": "auto",
	"compile": true
}
```

```shell
$ python src/train.py
```

---

> [!NOTE]
> You need to create `scripts`, `data`, `bin` and `res` folders.<br>
> Code for training the model and tokenizer was written by Andrej Karpathy:<br>
> 1. [minBPE repo](https://github.com/karpathy/minbpe/)<br>
> 2. [nanoGPT repo](https://github.com/karpathy/nanoGPT/)
