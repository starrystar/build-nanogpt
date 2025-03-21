import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention，当 is_causal=True 时，函数会自动生成一个下三角矩阵作为掩码
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class CausalSelfAttention1(nn.Module): # 视频里逐行实现scaled_dot_product_attention方法（看最早的提交）

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) = 1> torch.Size([5, 8, 768]) 2> torch.Size([5, 9, 768])
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)=torch.Size([5, 12, 8, 64])
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # 1> att.shape=torch.Size([5, 12, 8, 8]) 2> att.shape=torch.Size([5, 12, 9, 9])
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # 
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # tanh是Gelu的近似计算版本，用于加速
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention1(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # 即FFN

    def forward(self, x):
        # Trans DeBlock中Y=ln{X+drop[attn(X)]}。而下面的代码是GPT-2对Transformer的改进，梯度从输出开始计算就是一个加法，使得从top开始的梯度可以分为两个分支，残差分支的梯度直接从最后的输出到了输入，另一个分支则经过一系列复杂的层的变化到达输入。
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    # 注意！！！！
    # 注意！！！！
    # 注意！！！！
    # 本实现不包含在训练和eval时表现不同的modules和layers（如dropout、batchnorm或其他层）

    def __init__(self, config):
        super().__init__()
        self.config = config

        # nn.ModuleDict像是nn.ModuleList，可以使用对应的名字获取该子Module
        # 这里的wte、wpe等等都是对应到play.ipynb中的state_dict：
        #    transformer.wpe.weight torch.Size([1024, 768]) # position embedding, GPT-2的max_len=1024，即每个token最多可以关注1024个位置
        #    transformer.wte.weight torch.Size([50257, 768]) # weight of token embedding, bert中有self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        #    transformer.h.0.ln_1.weight torch.Size([768]) # 这里的h.0, h.1, ..., h.11表示12层Transformer
        #    transformer.h.0.ln_1.bias torch.Size([768])
        #    transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
        #    transformer.h.0.attn.c_attn.bias torch.Size([2304])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Transformer图中的OutputEmbedding，即TokenEmbedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # Transformer图Positional Encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # h[0]对应Transformer图DecoderBlock即灰色块
            ln_f = nn.LayerNorm(config.n_embd), # yuque中GPT2论文的最后一层是layernorm，是新添加的层
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Transformer图中Decoder最后的Linear。将Emb映射回vocab表，TransformerDecoder中有

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)，注意这里是在输入数据相同的设备上创建的，可以避免设备不一
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface
        下面的实现就是先用config参数初始化一个本文手写的model = GPT(config)，然后初始化一个从HuggingFace初始化的model_hf = GPT2LMHeadModel.from_pretrained(model_type)，之后对应参数的名字、去掉一些buffer参数掩码矩阵参数、对HuggingFace的Tensorflow实现中的一些参数做必要的转置，实现参数的复制。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param. 
        # .attn.bias是掩码下三角矩阵，仅用于自回归，这里忽略掉了
        # buffer指的是register_buffer（浏览器收藏夹）中的buffer，这里是说用于mask的那个下三角矩阵或者buffer中的参数，这是在旧的提交中写的（我放在了CausalSelfAttention的注释里）

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# create model
# model = GPT(GPTConfig(vocab_size=50304))
model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model.to(device)
model.eval()
num_return_sequences = 5
max_length = 32

import tiktoken
enc = tiktoken.get_encoding("gpt2") # tokenizer for gpt2
tokens = enc.encode("Hello, I'm a language model,") # tokenize，获得一个list of int，https://tiktokenizer.vercel.app/?model=gpt2
tokens = torch.tensor(tokens, dtype=torch.long) # shape=(8,) [15496, 11, 314, 1101, 257, 3303, 2746, 11]
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # shape=(5,8)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)
# (B,T)=(5,8)
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        # xgen.shape=torch.Size([5, 8])
        logits, loss = model(xgen) # -> logits.shape=torch.Size([B=5, T=8, vocab_size=50257])
        # 注意只保留seq最后位置的值，take the logits at the last position
        logits = logits[:, -1, :] # (B=5, vocab_size=50257)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50，是保留前50个概率，后面的都记为0并重新规范化。这是huggingface pipeline默认做的。
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # ix.shape=(B=5, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # xcol.shape=(B, 1). ix是(5,1)，就是保持5个的行idx不变，列idx替换为ix中的值。即xcol最后形状是(5,1)，为[ [(0,ix0)], [(1,ix1)], ..., [(4,ix4)] ]。因为ix约等于对这5行中各自最大的50个概率中取了一个最大的在[0,50)的索引，xcol就是对应原始50257的最大概率的索引。
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1) # 执行完这句，相当于把xcol=(5,1)在dim=1上concat到xgen原始输入上，xgen=(5,8)->(5,9)，就是把句子最后一个token对应的预测token，也就是预测句子接下来的一个token拼接到xgen上。
# print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens) # 这是tiktokenize中的，将token的id->原文。
    print(f"sample {i}: {decoded}")