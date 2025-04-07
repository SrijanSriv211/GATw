from shared.utils import calc_total_time
from colorama import init, Fore, Style
import pickle, regex, time

init(autoreset=True)

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# https://github.com/karpathy/minbpe/pull/82/files#diff-2f6d110dc37c6714f3b44335b029a950adfb0c58e2c3013e030a9bbdd76ed02d
def get_stats(ids, counts=None, weight=1):
	"""
	Given a list of integers, return a dictionary of counts of consecutive pairs, multiplied by weight
	Example: [1, 2, 3, 1, 2] -> {(1, 2): 2*weight, (2, 3): 1*weight, (3, 1): 1*weight}
	Optionally allows to update an existing dictionary of counts
	"""
	counts = {} if counts is None else counts
	for pair in zip(ids, ids[1:]): # iterate consecutive elements
		counts[pair] = counts.get(pair, 0) + weight
	return counts

def merge(ids, pair, idx):
	"""
	In the list of integers (ids), replace all consecutive occurrences
	of pair with the new integer token idx
	Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
	"""
	newids = []
	i = 0
	while i < len(ids):
		# if not at the very last position AND the pair matches, replace it
		if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
			newids.append(idx)
			i += 2

		else:
			newids.append(ids[i])
			i += 1

	return newids

class Encoder:
	def __init__(self, pattern=None):
		"""
		- pattern: optional string to override the default (GPT-4 split pattern)
		- special_tokens: str -> int dictionary of special tokens
			example: {'<|endoftext|>': 100257}
		"""
		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = regex.compile(self.pattern)
		self.special_tokens = {}
		self.inverse_special_tokens = {}
		self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

	def train(self, filename, vocab_size=256, text_range=None):
		"""
		- vocab_size: max number of merges to be made - 256 bytes
		- text_range: how much many chars from the text should be used to train (default: None means entire text)
		"""
		assert vocab_size >= 256
		start_time = time.time()

		with open(filename, "r", encoding="utf-8") as f:
			text = f.read()

		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		# split the text up into text chunks
		text_chunks = regex.findall(self.compiled_pattern, text if text_range < 0 else text[:text_range])
		del text

		print(f"encoding text chunks... {Fore.BLACK}{Style.BRIGHT}(takes a ~minute)")

		# input text preprocessing
		ids = [list(ch.encode("utf-8")) for ch in text_chunks]
		del text_chunks

		# keep just one instance of identical chunks, keep their count in idsw
		# https://github.com/karpathy/minbpe/pull/82/files#diff-6b5737d60acbc8d11dba46334d76c559796c1aca8d51e13ed069236f947b9e1f
		tmp = {}
		for byte_str in ids:
			byte_str = bytes(byte_str)
			tmp[byte_str] = tmp.get(byte_str, 0) + 1

		ids = [list(k) for k in map(list, tmp.keys())]
		idsw = list(tmp.values())

		print("training on vocab size", f"{Fore.WHITE}{Style.BRIGHT}{vocab_size}")
		last_print_time = time.time()

		# iteratively merge the most common pairs to create new tokens
		n_merges = vocab_size - 256
		for i in range(n_merges):
			# count the number of times every consecutive pair appears
			stats = {}

			# passing in stats will update it in place, adding up counts
			for j, chunk_ids in enumerate(ids):
				get_stats(chunk_ids, stats, idsw[j])

			# find the pair with the highest count
			pair = max(stats, key=stats.get)

			# mint a new token: assign it the next available id
			idx = 256 + i
			self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

			# replace all occurrences of pair in ids with idx
			ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

			# verbose
			current_print_time = time.time()
			print(
				f"{Fore.WHITE}{Style.BRIGHT}merge",
				f"{Fore.BLACK}{Style.BRIGHT}[{i+1}/{n_merges}]"
				":",
				f"{pair} -> {idx}",
				f"{Fore.WHITE}{Style.DIM}({self.vocab[idx]})",
				f"had {Fore.WHITE}{Style.BRIGHT}{stats[pair]}{Style.RESET_ALL} occurrences"
				f"{Style.RESET_ALL},",
				f"{Fore.BLACK}{Style.BRIGHT}time taken: {calc_total_time(current_print_time-last_print_time)}"
			)
			last_print_time = current_print_time

		# print the total time taken to do all the merges
		print("vocab size:", f"{Fore.WHITE}{Style.BRIGHT}{len(self.vocab)}")
		print("time taken:", f"{Fore.WHITE}{Style.BRIGHT}{calc_total_time(time.time()-start_time)}")

	# special_tokens is a dictionary of str -> int
	# example: {"<|endoftext|>": 100257}
	def register_special_tokens(self, *special_tokens):
		self.special_tokens = dict([(x, i + len(self.vocab)) for i, x in enumerate(special_tokens)])
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

	# given ids (list of integers), return Python string
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

	# return the token ids
	# https://github.com/karpathy/minbpe/pull/84/files#diff-6b5737d60acbc8d11dba46334d76c559796c1aca8d51e13ed069236f947b9e1f
	def _encode_chunk(self, text_bytes, inverse_vocab):
		n = len(text_bytes)
		dp = [float("inf") for _ in range(n)]
		backtrack = [[-1] * n for _ in range(n)]

		# at i'th iteration, dp[j] is the size of optimal tokenization of text_bytes[:j+1], for 0 < j < i-1, and
		# backtrack[j] is the index of the first byte of the last token of an optimal tokenization.
		for i in range(0, n):
			if text_bytes[:i+1] in inverse_vocab:
				dp[i] = 1
				backtrack[i] = 0
				continue
			for j in range(1, i+1):
				if text_bytes[j:i+1] in inverse_vocab:
					assert dp[j-1] >= 1
					if dp[i] > dp[j-1] + 1:
						dp[i] = dp[j-1] + 1
						backtrack[i] = j

		# reconstruct the tokens from the backtrack table
		def reconstruct_tokens(i):
			if backtrack[i] == 0:
				return [text_bytes[:i+1]]
			k = backtrack[i]
			return reconstruct_tokens(k-1) + [text_bytes[k:i+1]]

		return [inverse_vocab[token] for token in reconstruct_tokens(n - 1)]

	def encode_ordinary(self, text, inverse_vocab):
		"""Encoding that ignores any special tokens."""
		cache = {}

		# split text into chunks of text by categories defined in regex pattern
		text_chunks = regex.findall(self.compiled_pattern, text)

		# all chunks of text are encoded separately, then results are joined
		ids = []
		for chunk in text_chunks:
			if chunk not in cache:
				chunk_bytes = chunk.encode("utf-8")  # raw bytes
				cache[chunk] = self._encode_chunk(chunk_bytes, inverse_vocab)
			ids.extend(cache[chunk])
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

		inverse_vocab = {v: k for k, v in self.vocab.items()}
		# shortcut: if no special tokens, just use the ordinary encoding
		if not special:
			return self.encode_ordinary(text, inverse_vocab)

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
				ids.extend(self.encode_ordinary(part, inverse_vocab))

		return ids

	def save(self, checkpoint):
		"""
		Saves two files: checkpoint.bin
		- model file is the critical one, intended for load()
		"""
		# write the model: to be used in load() later
		with open(checkpoint, "wb") as f:
			pickle.dump({
				"pattern": self.pattern,
				"special": self.special_tokens,
				"vocab": self.vocab
			}, f)

	def load(self, checkpoint: str):
		# read the model file
		with open(checkpoint, "rb") as f:
			model = pickle.load(f)

		self.pattern = model["pattern"]
		self.special_tokens = model["special"]
		self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
		self.vocab = model["vocab"]
