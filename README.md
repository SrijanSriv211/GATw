# GATw
## Generative Alphabet Transformers write
A super small and simple language model

> [!NOTE]
> You need to create `scripts`, `data`, `bin` and `res` folders.<br>
> Code for training the model and tokenizer was written by Andrej Karpathy:<br>
> 1. [minBPE repo](https://github.com/karpathy/minbpe/)<br>
> 2. [nanoGPT repo](https://github.com/karpathy/nanoGPT/)

This is how `scripts\config.json` should look-like

```json
{
	"load_from_file": true,
	"train_data": "bin\\train.bin",
	"val_data": "bin\\val.bin",
	"init_from": "scratch",

	"checkpoints": {
		"path": "bin\\checkpoints",
		"interval": 100
	},
	"save_path": "bin\\GATw.bin",

	"max_iters": 10000,
	"eval_interval": 100,
	"log_interval": 10,
	"eval_iters": 100,
	"encoder_path": "bin\\cl4k.bin",
	"gen_interval": 500,
	"gen_iters": 3,

	"gradient_accumulation_steps": 4,
	"batch_size": 16,
	"block_size": 256,

	"vocab_size": 4096,
	"n_layer": 6,
	"n_head": 6,
	"n_embd": 96,
	"n_hidden": "4x_embd",
	"dropout": 0,

	"learning_rate": 3e-4,
	"weight_decay": 0.1,
	"grad_clip": 1,

	"decay_lr": true,
	"warmup_iters": 100,
	"lr_decay_iters": 10000,
	"min_lr": 3e-5,
	"beta1": 0.9,
	"beta2": 0.95,

	"device": "cpu",
	"seed": "auto",
	"compile": true
}
```


This is how `scripts\enc_config.json` should look like:
```json
{
	"dataset_path": "data\\data.txt",
	"merge_vocab_size": 2043,
	"text_range": 50000000,
	"special_tokens": ["<|sot|>", "<|eot|>", "<|pad|>", "<|sep|>", "<|reason|>"],
	"outpath": "bin\\cl2k.bin"
}
```


And this is how `scripts\prep_data_config.json` should look like:
```json
{
	"enc_path": "bin\\cl2k.bin",
	"dataset_path": "data\\base",
	"outpath": "bin",
	"data_division": 0.8,
	"distribution": null
}
```
