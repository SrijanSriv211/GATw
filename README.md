# GATw
A super small and simple language model

This is how `config.json` should look-like

```json
{
	"load_from_file": true,
	"train_data": "bin\\train.bin",
	"val_data": "bin\\val.bin",
	"init_from": "scratch",

	"checkpoints": {
		"path": "bin\\checkpoints",
		"interval": 500
	},
	"save_path": "bin\\GATw.bin",

	"max_iters": 150000,
	"eval_interval": 500,
	"log_interval": 20,
	"eval_iters": 500,

	"gradient_accumulation_steps": 1,
	"batch_size": 8,
	"block_size": 256,

	"vocab_size": 1225,
	"n_layer": 4,
	"n_head": 8,
	"n_embd": 192,
	"n_hidden": 576,
	"dropout": 0,

	"learning_rate": 3e-4,
	"weight_decay": 0.1,
	"grad_clip": 1,

	"decay_lr": true,
	"warmup_iters": 100,
	"lr_decay_iters": 150000,
	"min_lr": 1e-5,
	"beta1": 0.9,
	"beta2": 0.95,

	"device": "cpu",
	"seed": "auto",
	"compile": true
}
```


This is how `enc_config.json` should look like:
```json
{
	"dataset_path": "data\\data.txt",
	"merge_vocab_size": 2043,
	"text_range": 50000000,
	"special_tokens": ["<|sot|>", "<|eot|>", "<|pad|>", "<|sep|>", "<|reason|>"],
	"outpath": "bin\\cl2k.bin"
}
```


And this is how `prep_data_config.json` should look like:
```json
{
	"enc_path": "bin\\cl2k.bin",
	"dataset_path": "data\\base",
	"outpath": "bin",
	"data_division": 0.8,
	"distribution": null
}
```
