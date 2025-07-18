from colorama import Style, Fore, init
from encoder.bytepair import Encoder
from models.gpt import sample
import warnings, torch, sys

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
init(autoreset = True)

enc = Encoder()
enc.load("bin/cl8k.bin")

s = sample(enc=enc)
s.load(torch.load(sys.argv[1]), True)

test = [
	"Google",
	"Shah Rukh Khan",
	"Hello"
]

for txt in test:
	enctxt = enc.encode(txt, allowed_special="all") if txt != None else txt
	out = enc.decode(s.generate(enctxt, length=256, temperature=0.8))
	print(f"{Fore.WHITE}{Style.DIM}```\n{out}\n```\n")
