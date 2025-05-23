from colorama import Style, Fore, init
from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch, inspect, math

init(autoreset=True)

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        self.rope_base = config.rope_base

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print(f"{Fore.RED}{Style.BRIGHT}WARNING: {Fore.WHITE}{Style.DIM}using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

        # https://github.com/karpathy/nanoGPT/pull/590/files
        if self.use_rope:
            hs = self.n_embd // self.n_head
            d = hs // 2
            self.register_buffer('theta', 1.0 / (self.rope_base ** (2 * torch.arange(0, d, dtype=torch.float32) / hs)))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.use_rope:
            # Use precomputed theta and cast to input dtype
            hs = C // self.n_head  # head dimension
            d = hs // 2  # number of dimension pairs
            theta = self.theta[:d].to(x.dtype)  # Cast to input dtype

            # Compute frequencies using input dtype
            t = torch.arange(T, device=x.device, dtype=x.dtype)
            freqs = torch.outer(t, theta)
            freqs_cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
            freqs_sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)

            # Optimized RoPE application without type casting
            def apply_rope(x, cos, sin):
                x_ = x.reshape(*x.shape[:-1], -1, 2)  # (B, nh, T, d, 2)
                x0 = x_[..., 0]
                x1 = x_[..., 1]
                x0_rot = x0 * cos - x1 * sin
                x1_rot = x0 * sin + x1 * cos
                x_rot = torch.stack([x0_rot, x1_rot], dim=-1).flatten(start_dim=-2)
                return x_rot

            q = apply_rope(q, freqs_cos, freqs_sin)
            k = apply_rope(k, freqs_cos, freqs_sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_hidden, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 4282
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    n_hidden: int = 32 * 4 # feedforward hidden size. Typically is set to `4 * n_embd`
    beta1: float = 0.9
    beta2: float = 0.95
    dropout: float = 0.0
    use_rope: bool = False
    rope_base: float = 10000.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.use_rope = config.use_rope

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = None if self.use_rope else nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.use_rope:
            x = self.transformer.drop(tok_emb)

        else:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        if not self.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {Fore.WHITE}{Style.BRIGHT}{len(decay_params)}"
            f"{Style.RESET_ALL},",
            f"with {Fore.WHITE}{Style.BRIGHT}{num_decay_params:,}",
            "parameters"
        )

        print(
            f"num non-decayed parameter tensors: {Fore.WHITE}{Style.BRIGHT}{len(nodecay_params)}"
            f"{Style.RESET_ALL},",
            f"with {Fore.WHITE}{Style.BRIGHT}{num_nodecay_params:,}",
            "parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        color = f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}" if use_fused == True else f"{Fore.LIGHTRED_EX}{Style.BRIGHT}"
        print(f"using fused AdamW: {color}{use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(self.config.beta1, self.config.beta2), fused=use_fused)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            # https://github.com/karpathy/nanoGPT/pull/546/
            if temperature == 0:
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            else:
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#                    _____         __  __ _____  _      ______ 
#                   / ____|  /\   |  \/  |  __ \| |    |  ____|
#                  | (___   /  \  | \  / | |__) | |    | |__   
#                   \___ \ / /\ \ | |\/| |  ___/| |    |  __|  
#                   ____) / ____ \| |  | | |    | |____| |____ 
#                  |_____/_/    \_\_|  |_|_|    |______|______|
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class sample:
    def __init__(self, device="auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def load(self, checkpoint, compile=False):
        # create an instance of GPT
        gptconf = GPTConfig(**checkpoint["hyperparams"])
        self.model = GPT(gptconf)

        # remove `_orig_mod.` prefix from state_dict (if it's there)
        state_dict = checkpoint["model"]
        unwanted_prefix = '_orig_mod.'

        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # load the saved model state_dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() # set the model to evaluation mode

        if compile: self.model = torch.compile(self.model)

    # use the model for generation or other tasks
    def generate(self, encoded_text=None, length=1024, temperature=1, top_k=None):
        """
        `max_new_tokens`: number of tokens generated in each sample
        `temperature`: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        `tok_k`: retain only the top_k most likely tokens, clamp others to have 0 probability
        """

        return self.model.generate(self.prepare_context(encoded_text), max_new_tokens=length, temperature=temperature, top_k=top_k)[0].tolist()

    def prepare_context(self, encoded_text):
        if encoded_text == None:
            return torch.zeros((1, 1), dtype=torch.long, device=self.device)

        return torch.tensor(encoded_text, dtype=torch.long, device=self.device).unsqueeze(0)
