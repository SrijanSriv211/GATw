from encoder.bytepair import Encoder
enc = Encoder()

#* set `vocab_size` in `config.json` 1225 (35 squared)
enc.train("data\\data.txt", 1220, text_range=50_000_000)
enc.register_special_tokens("<|sot|>", "<|eot|>", "<|pad|>", "<|sep|>", "<|reason|>")
enc.save("bin\\cl1k.bin")
