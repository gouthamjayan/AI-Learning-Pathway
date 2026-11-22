# Transformer (MiniTransformer)

A **Transformer** is a sequence-modeling architecture built around **self-attention** rather than recurrence or convolution, allowing tokens across a sequence to exchange context directly in parallel[cite: 7]. In `transformer.py`, `MiniTransformer` implements a decoder-style language modeling architecture using **Pre-LN** normalization, **Multi-Head Attention**, **GELU feed-forward layers**, **residual connections**, and **learned positional embeddings**[cite: 7].

---

## Core Intuition & Architecture

### 1. Token & Positional Embeddings (`MiniTransformer`)
Attention itself is permutation-aware only through positional information[cite: 7]. Given token IDs of shape `(batch_size, seq_len)`, `MiniTransformer` maps both token identity and token position into `d_model`-dimensional vectors and adds them[cite: 7]:

$$X_0 = \text{TokenEmbedding}(\text{idx}) + \text{PosEmbedding}(\text{positions})$$

* **`token_embedding` (`nn.Embedding(vocab_size, d_model)`):** Maps discrete vocabulary indices into dense embeddings of width `d_model`[cite: 7].
* **`pos_embedding` (`nn.Embedding(max_seq_len, d_model)`):** Encodes sequence positions `[0, 1, ..., seq_len - 1]` up to `max_seq_len`[cite: 7].

### 2. Transformer Block (`TransformerBlock`)
Each block repeats two sub-layers—**Attention** (token-to-token communication) and **MLP** (per-token feature processing)—using a **Pre-LayerNorm (Pre-LN)** layout with **residual connections**[cite: 7]:

1. **Attention sub-layer:**
   $$X' = X + \text{MultiHeadAttention}(\text{LayerNorm}_1(X))$$
   In `transformer.py`, `nn.MultiheadAttention(..., batch_first=True)` takes `(normed_x, normed_x, normed_x)` as query, key, and value so each position attends across the sequence[cite: 7].
2. **Feed-Forward / MLP sub-layer:**
   $$X_{\text{out}} = X' + \text{MLP}(\text{LayerNorm}_2(X'))$$
   The position-wise MLP expands from `d_model` to `d_ff`, applies `nn.GELU()`, uses `nn.Dropout(dropout)`, and projects back to `d_model`[cite: 7].

### 3. Final Norm & Language Modeling Head (`MiniTransformer`)
After passing through `num_layers` stacked `TransformerBlock` modules (`nn.ModuleList`), the sequence passes through:

* **`norm_f` (`nn.LayerNorm(d_model)`):** Final layer normalization across the hidden dimension[cite: 7].
* **`lm_head` (`nn.Linear(d_model, vocab_size)`):** Projects each position's hidden vector from `d_model` back to `vocab_size` logits, producing shape `(batch_size, seq_len, vocab_size)`[cite: 7].

---

## Key Practical Considerations

* **Pre-LN vs. Post-LN & Residual Streams:** Normalizing *before* attention and MLP (`Pre-LN`) and adding the un-normed input back via residual connections (`x = x + ...`) keeps gradient flow cleaner across deeper stacks[cite: 7].
* **Shape Flow Through the Network:**
  * Input IDs: `(batch_size, seq_len)`[cite: 7]
  * Embeddings & Transformer blocks: `(batch_size, seq_len, d_model)`[cite: 7]
  * Output logits: `(batch_size, seq_len, vocab_size)`[cite: 7]
* **Sequence Length Limit:** Because `pos_embedding` uses `nn.Embedding(max_seq_len, d_model)`, `seq_len` cannot exceed `max_seq_len` without raising an out-of-bounds index error[cite: 7].
* **Language Modeling Objective:** `logits` represent unnormalized next-token/token-level scores over `vocab_size` (typically paired with cross-entropy loss during training)[cite: 7].

---

## Files in This Module

* `transformer.py` — PyTorch implementation of `TransformerBlock` and `MiniTransformer`, including a shape-verification test script (`vocab_size=5000`, `d_model=256`, `num_heads=8`, `num_layers=4`, `d_ff=1024`, `max_seq_len=128`)[cite: 7].