from shared.utils import calc_total_time
from colorama import init, Fore, Style
import pickle, regex, time

init(autoreset=True)

class Encoder:
	def __init__(self):
		"""
		- pattern: optional string to override the default (GPT-4 split pattern)
		- special_tokens: str -> int dictionary of special tokens
			example: {'<|endoftext|>': 100257}
		"""

		self.special_tokens = {}
		self.inverse_special_tokens = {}
		# base vocab
		self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

	def train(self, filename):
		start_time = time.time()

		with open(filename, "r", encoding="utf-8") as f:
			text = f.read()

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		# here are all the unique char that occur in this text
		f256_chr = [chr(idx) for idx in range(256)] # first 256 chars
		t_chr = sorted(list(set(list(text) + f256_chr))) # all text chars
		del f256_chr, text

		self.vocab = {i: ch.encode("utf-8") for i, ch in enumerate(t_chr)} # char -> idx

		# print the total time taken to do all the merges
		print("vocab size:", f"{Fore.WHITE}{Style.BRIGHT}{len(self.vocab)}")
		print("time taken:", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, *special_tokens):
		self.special_tokens = dict([(x, i + len(self.vocab)) for i, x in enumerate(special_tokens)])
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

	def decode(self, ids):
		part_bytes = []
		for idx in ids:
			if idx in self.vocab:
				part_bytes.append(self.vocab[idx])

			elif idx in self.inverse_special_tokens:
				part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))

			else:
				raise ValueError(f"invalid token id: {idx}")

		text_bytes = b"".join(part_bytes)
		return text_bytes.decode("utf-8", errors="replace")

	def encode_ordinary(self, text):
		"""Encoding that ignores any special tokens."""
		inverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
		ids = []

		for i in text:
			if i.encode("utf-8") in self.vocab.values():
				ids.append(inverse_vocab[i.encode("utf-8")])

		return ids

	def encode(self, text, allowed_special="none_raise"):
		"""
		Unlike encode_ordinary, this function handles special tokens.
		allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
		if none_raise, then an error is raised if any special token is encountered in text
		this is the default tiktoken behavior right now as well
		any other behavior is either annoying, or a major footgun
		"""
		# decode the user desire w.r.t. handling of special tokens
		special = None
		if allowed_special == "all":
			special = self.special_tokens

		elif allowed_special == "none":
			special = {}

		elif allowed_special == "none_raise":
			special = {}
			assert all(token not in text for token in self.special_tokens)

		elif isinstance(allowed_special, set):
			special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

		else:
			raise ValueError(f"allowed_special={allowed_special} not understood")

		# shortcut: if no special tokens, just use the ordinary encoding
		if not special:
			return self.encode_ordinary(text)

		# otherwise, we have to be careful with potential special tokens in text
		# we handle special tokens by splitting the text
		# based on the occurrence of any exact match with any of the special tokens
		# we can use regex.split for this. note that surrounding the pattern with ()
		# makes it into a capturing group, so the special tokens will be included
		special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
		special_chunks = regex.split(special_pattern, text)
	
		# now all the special characters are separated from the rest of the text
		# all chunks of text are encoded separately, then results are joined
		ids = []
		for part in special_chunks:
			if part in special:
				# this is a special token, encode it separately as a special case
				ids.append(special[part])

			else:
				# this is an ordinary sequence, encode it normally
				ids.extend(self.encode_ordinary(part))

		return ids

	def save(self, checkpoint):
		"""
		Saves two files: checkpoint.bin
		- model file is the critical one, intended for load()
		"""
		# write the model: to be used in load() later
		with open(checkpoint, "wb") as f:
			pickle.dump({
				"special": self.special_tokens,
				"vocab": self.vocab
			}, f)

	def load(self, checkpoint: str):
		# read the model file
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)

		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]
