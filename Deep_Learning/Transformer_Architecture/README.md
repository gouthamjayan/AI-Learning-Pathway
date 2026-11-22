# Transformer

A Transformer processes sequences of tokens and lets them pay attention to one another to build context across the input[cite: 5].

## How It Works
- Turn token IDs and token positions into vectors and add them together[cite: 5].
- Pass the sequence through stacked `TransformerBlock` layers, where attention lets tokens share context and a feed-forward (`mlp`) block with `GELU` processes each token's features, using normalization and residual connections along the way[cite: 5].
- Apply a final normalization layer and project to vocabulary logits (`lm_head`)[cite: 5].

## What's in This Folder
- **`transformer.py`** — Implements `TransformerBlock` and `MiniTransformer` in PyTorch along with a test script that checks input and output logits shapes[cite: 5].

## Good Things to Remember
- Keep sequence length within `max_seq_len` so position embeddings stay in bounds[cite: 5].
- Check tensor shapes as they move from `(batch_size, seq_len)` to `(batch_size, seq_len, d_model)` and finally `(batch_size, seq_len, vocab_size)`[cite: 5].