import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections for Q, K, V
        Q = self.q_linear(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.out_linear(attn_output)  # (batch_size, seq_len, embed_dim)

        return output

# Example usage
batch_size = 2
seq_len = 10
embed_dim = 512
num_heads = 4

query = torch.rand(batch_size, seq_len, embed_dim)
key = torch.rand(batch_size, seq_len, embed_dim)
value = torch.rand(batch_size, seq_len, embed_dim)

multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
output = multi_head_attention(query, key, value)

print("Output shape:", output.shape)  # Should be (batch_size, seq_len, embed_dim)
