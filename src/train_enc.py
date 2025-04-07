from encoder.bytepair import Encoder
enc = Encoder()

#* set `vocab_size` in `config.json` 5184
enc.train("data\\data.txt", 5180, text_range=70_000_000)
enc.register_special_tokens("<|sot|>", "<|eot|>", "<|pad|>", "<|sep|>")
enc.save("bin\\cl5k.bin")
