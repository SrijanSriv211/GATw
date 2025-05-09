from encoder.bytepair import Encoder
import json, sys
enc = Encoder()

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "scripts/enc_config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	CONFIG = json.load(f)

#* set `vocab_size` in `config.json` 2048
enc.train(CONFIG["dataset_path"], CONFIG["merge_vocab_size"], text_range=CONFIG["text_range"])
enc.register_special_tokens(*CONFIG["special_tokens"])
enc.save(CONFIG["outpath"])
