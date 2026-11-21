import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Normalization layers (Pre-LN architecture, standard in modern LLMs)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 2. The "Tongue": Multi-Head Attention (Dynamic context routing)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 3. The "Brain": Position-wise Feed-Forward Network (MLP / Knowledge store)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # Non-linear activation function
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # --- Step 1: Attention (Communication) with Residual Connection ---
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + attn_output  # Add original input back (residual connection)
        
        # --- Step 2: MLP (Processing) with Residual Connection ---
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output   # Add original input back (residual connection)
        
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Input Embeddings: Maps Token IDs and Positions into d_model vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 2. Stack of Transformer Blocks (Building depth by repeating the floor)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # 3. Final Normalization and LM Head (Unembedding back to vocabulary logits)
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        batch_size, seq_len = idx.size()
        
        # Generate position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Step A: Combine what the word is + where the word is located
        x = self.token_embedding(idx) + self.pos_embedding(positions)
        
        # Step B: Push the vectors through the stack of Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Step C: Final normalization
        x = self.norm_f(x)
        
        # Step D: Project back to vocabulary size to get prediction logits for every token
        logits = self.lm_head(x)
        
        return logits


# --- Test Script: Run this to verify shapes ---
if __name__ == "__main__":
    vocab_size = 5000   # Size of our dictionary of words/subwords
    d_model = 256       # Width of every feature vector
    num_heads = 8       # Number of attention heads in the "tongue"
    num_layers = 4      # Number of stacked transformer blocks (floors)
    d_ff = 1024         # Hidden dimension expansion inside the MLP brain
    max_seq_len = 128   # Maximum sentence length supported

    # Instantiate model
    model = MiniTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)

    # Mock input: Batch of 2 sentences, each containing 16 token IDs
    sample_input = torch.randint(0, vocab_size, (2, 16))

    # Forward pass
    logits = model(sample_input)

    print("--- Model Execution Successful ---")
    print(f"Input shape (Batch, Seq_Len): {sample_input.shape}")
    print(f"Output logits shape (Batch, Seq_Len, Vocab_Size): {logits.shape}")