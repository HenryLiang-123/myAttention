import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # embed_size = size of word embedding vector
        # heads = # of attention blocks
        # head_dim = output size of 1 attention block
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads # Distribute size of the embedding evenly across the heads

        # Embedding size needs to be divisible by heads
        assert (self.head_dim * heads == embed_size)

        # FC layers 
        # Connect each (word embedding + positional encoding) to each neuron in Q, K, V
        # Input of (word embedding + positional encoding) is linearly projected into Q, K, V for each head
        # Attention mechanism is applied to each head, which results in 1 output per head 
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # Combines the output of the heads to return to the size of the embeddings

    def forward(self, values, keys, query, mask):
        '''
        values: [N, value_len, features]. N is the batch size, value_len is the sequence length, features is the total number of features (embed_size)
        keys: [N, key_len, features]. N is the batch size, key_len is the sequence length, features is the total number of features (embed_size)
        query: [N, query_len, features]. N is the batch size, query_len is the sequence length, features is the total number of features (embed_size)
        '''
        N = query.shape[0] # Number of training examples or batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embed_size into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Multiplying keys and queries
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # n: batch size, q: query_len, h: heads, d: head_dim, k: key_len
        # values shape [N, value_len, heads, head_dim]
        # keys shape [N, key_len, heads, head_dim]
        # queries shape [N, query_len, heads, head_dim]
        # energy shape [N, heads, query_len, key_len]
        """
        Following code works the same
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)

        # Transpose the keys for matrix multiplication to shape [batch_size, num_heads, head_dim, seq_length]
        key_transposed = key.transpose(-2, -1)

        # Perform matrix multiplication resulting in shape [batch_size, num_heads, query_len, key_len]
        energy = torch.matmul(query, key_transposed)
        """

        if mask is not None:
            # Makes the model focus on the relevant parts of the input
            # Maintain autoregressive property of decoder
            # Ensure each output token is conditioned only oin previous output tokens
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        ) # l is the dimension to multiply across
        # attention shape [N, heads, query_len, key_len]
        # values shape [N, value_len, heads, head_dim]
        # after einsim [N, query_len, heads, head_dim], then flatten last 2 dims

        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads) # self attention block
        self.norm1 = nn.LayerNorm(embed_size) # Normalize outputs
        self.norm2 = nn.LayerNorm(embed_size) # Normalize outputs

        # Further transforms the output from the self attention layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



