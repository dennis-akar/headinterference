# %%

"""
https://docs.google.com/document/d/1n4h7tlNEn5xHfoDsRv5K6mpLkKRjYsBUtTXS7-KpchA

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 as tokens.

11 as bos.

assume length is always the same.

identity matrix as embedding and unembedding.

5 attention heads, one layer. (probably superposition)

12 basis residual stream.
"""

# restart kernel requirement

pls_restart = False
if "pls_restart" in globals() and eval("pls_restart") or "pls_restart_after_one_run" in globals() and eval("pls_restart_after_one_run"):
    print("i kill session" )
    print("now i die bye have a nice day")
    exit()
pls_restart_after_one_run = False

# imports

import sys, os, subprocess
from functools import partial
sub_run = partial(subprocess.run, shell=True, check=True)
try:
    import einops, jaxtyping, transformer_lens, circuitsvis
except ModuleNotFoundError:
    sub_run = partial(subprocess.run, shell=True, check=True)
    sub_run("pip install einops")
    sub_run("pip install jaxtyping")
    sub_run("pip install transformer_lens")
    sub_run("pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python")
try:
    import ioi_dataset, path_patching
except ModuleNotFoundError:
    print("Downloading and installing path patching files in", os.getcwd())
    sub_run("wget https://github.com/callummcdougall/path_patching/archive/refs/heads/main.zip")
    sub_run("unzip main.zip 'path_patching-main/ioi_dataset.py'")
    sub_run("unzip main.zip 'path_patching-main/path_patching.py'")
    sys.path.append("path_patching-main")
    os.remove("main.zip")
    os.rename("path_patching-main/ioi_dataset.py", "ioi_dataset.py")
    os.rename("path_patching-main/path_patching.py", "path_patching.py")
    os.rmdir("path_patching-main")


from dataclasses import dataclass
import math
from tqdm import tqdm
import plotly.express as px
import numpy as np

import torch as t
import torch.nn as nn
t.manual_seed(0)

from transformer_lens.utils import get_corner, gelu_new, tokenize_and_concatenate

import einops as eo
from einops import einsum as es

from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from torch import Tensor
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


# dataset


@dataclass(frozen=True)
class dataset:

    # config

    number_of_prompts = 64 * 64
    fraction_of_skip_trigrams = 0.3
    assert 0 <= fraction_of_skip_trigrams <= 1
    length_of_prompts = 5
    length_of_prompts = 5
    assert length_of_prompts >= 3

    skip_trigram_attend_from_token = t.tensor([0])
    skip_trigram_attend_to_tokens = t.tensor([i for i in range(1, 6)])
    skip_trigram_output_tokens = t.tensor([i for i in range(6, 11)])
    other_tokens = t.tensor([])
    bos_token = t.tensor([11])

    # process

    skip_trigram_count = skip_trigram_attend_to_tokens.shape[0]
    assert skip_trigram_attend_to_tokens.ndim == skip_trigram_output_tokens.ndim == 1
    assert skip_trigram_count == skip_trigram_output_tokens.shape[0]

    non_bos_tokens = t.concat((skip_trigram_attend_from_token, skip_trigram_attend_to_tokens, skip_trigram_output_tokens, other_tokens))

    tokens = t.concat((non_bos_tokens, bos_token))

    skip_trigrams = t.empty((skip_trigram_count, 3), dtype=t.long)
    skip_trigrams[:, 0] = skip_trigram_attend_from_token
    skip_trigrams[:, 1] = skip_trigram_attend_to_tokens
    skip_trigrams[:, 2] = skip_trigram_output_tokens

    data = t.full((number_of_prompts, length_of_prompts), bos_token.item())
    data[:, 1:] = non_bos_tokens[t.randint(len(non_bos_tokens), (number_of_prompts, length_of_prompts-1))]
    
    for i in range(number_of_prompts):
        prob = t.rand(1)
        if prob < fraction_of_skip_trigrams:
            data[i, (1, -2, -1)] = skip_trigrams[int(prob * skip_trigram_count), :]
    del prob, i
dataset = dataset()


# transformer


@dataclass(frozen=True)
class transformer_cfg:
    d_model = 12
    assert type(d_model) == int and d_model > 0
    
    debug = True
    assert type(debug) == bool
    
    layer_norm_eps = 1e-5
    assert type(layer_norm_eps) == float and layer_norm_eps > 0.0
    
    d_vocab = len(dataset.tokens)
    assert type(d_vocab) == int and d_vocab > 0

    embed_unembed_eye = True
    assert type(embed_unembed_eye) == bool
    if embed_unembed_eye:
        assert d_vocab <= d_model
    
    init_range = 0.02
    assert type(init_range) == float and layer_norm_eps > 0.0
    
    n_ctx = dataset.length_of_prompts
    assert type(n_ctx) == int and n_ctx > 0
    
    d_head = 3
    assert type(d_head) == int and d_head >= 0
    
    d_mlp = 0
    assert type(d_mlp) == int and d_mlp >= 0
    
    n_heads = 5
    assert type(n_heads) == int and n_heads >= 0
    
    n_layers = 1
    assert type(n_layers) == int and n_layers >= 0
transformer_cfg = transformer_cfg()

# @dataclass(frozen=True)
# class transformer_cfg:
#     d_model: int = 768
#     assert d_model > 0
#     debug: bool = True
#     layer_norm_eps: float = 1e-5
#     d_vocab: int = 50257
#     init_range: float = 0.02
#     n_ctx: int = 1024
#     d_head: int = 64
#     d_mlp: int = 3072
#     n_heads: int = 12
#     n_layers: int = 12
# transformer_cfg = transformer_cfg()


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.embed_unembed_eye:
            assert cfg.d_vocab <= cfg.d_model
            self.W_E = t.eye(cfg.d_model).to(device)
        else:
            self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
            nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return eo.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = es(
            normalized_resid_pre, self.W_Q,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
        ) + self.b_Q
        k = es(
            normalized_resid_pre, self.W_K,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
        ) + self.b_K
        v = es(
            normalized_resid_pre, self.W_V,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
        ) + self.b_V

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = es(
            q, k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K", 
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = es(
            v, attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head", 
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = es(
            z, self.W_O,
            "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model", 
        ) + self.b_O

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = es(
            normalized_resid_mid, self.W_in,
            "batch position d_model, d_model d_mlp -> batch position d_mlp", 
        ) + self.b_in
        post = gelu_new(pre)
        mlp_out = es(
            post, self.W_out,
            "batch position d_mlp, d_mlp d_model -> batch position d_model", 
        ) + self.b_out
        return mlp_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.d_head > 0 and cfg.n_heads > 0:
            self.ln1 = LayerNorm(cfg)
            self.attn = Attention(cfg)
        if cfg.d_mlp > 0:
            self.ln2 = LayerNorm(cfg)
            self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        if self.cfg.d_head > 0 and self.cfg.n_heads > 0:
            resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        else:
            resid_mid = resid_pre
        if self.cfg.d_mlp > 0:
            resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        else:
            resid_post = resid_mid
        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.embed_unembed_eye:
            assert cfg.d_vocab <= cfg.d_model
            self.W_U = t.eye(cfg.d_model).to(device)
            self.b_U = t.Tensor([0]).to(device)
        else:
            self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
            nn.init.normal_(self.W_U, std=self.cfg.init_range)
            self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return es(
            normalized_resid_final, self.W_U,
            "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
        ) + self.b_U
        # Or, could just do `normalized_resid_final @ self.W_U + self.b_U`


class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits


# train


@dataclass(frozen=True)
class training_cfg():
    batch_size = 64 # 8
    num_epochs = 1
    max_steps = float("inf")
    log_every = 100
    lr = 1e-3
    weight_decay = 1e-2
training_cfg = training_cfg()


# def get_log_probs(
# 	logits: Float[Tensor, "batch posn d_vocab"], 
# 	tokens: Int[Tensor, "batch posn"]
# ) -> Float[Tensor, "batch posn-1"]:
    
# 	log_probs = logits.log_softmax(dim=-1)
# 	# Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
# 	log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# 	return log_probs_for_tokens

def lm_cross_entropy_loss(logits, tokens):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()


# pred_log_probs = get_log_probs(demo_logits, tokens)
# print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
# print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
# print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
# print(dataset)
# print(dataset[0]['text'][:100])

model = DemoTransformer(transformer_cfg)
model.to(device)

optimizer = t.optim.AdamW(model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)

dataset_batched = eo.rearrange(dataset.data, "(batch number_of_prompts) len_prompts -> batch number_of_prompts len_prompts", batch=training_cfg.batch_size)

losses = []
print("Number of batches:", dataset.data.shape[0])
for epoch in range(training_cfg.num_epochs):
    for c, batch in tqdm(enumerate(dataset.data)):
        # print(batch.shape)
        # print(batch.unsqueeze(0))
        tokens = batch.unsqueeze(0).to(device)#['tokens'].cuda()
        logits = model(tokens)
        loss = lm_cross_entropy_loss(logits, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if c % training_cfg.log_every == 0:
            print(f"Step: {c}, Loss: {loss.item():.4f}")
        if c > training_cfg.max_steps:
            break

px.line(y=losses, x=np.arange(len(losses))*(transformer_cfg.n_ctx * training_cfg.batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")

# %%
# analyze

px.line(y=losses, x=np.arange(len(losses))*(transformer_cfg.n_ctx * training_cfg.batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")


