import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------
# MultiHeadAttention
# ------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads  # cần đảm bảo d_model chia hết cho n_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear   = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear   = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # x có shape [batch, seq_len, d_model]
        # chuyển thành [batch, seq_len, n_heads, depth] rồi permute thành [batch, n_heads, seq_len, depth]
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.split_heads(self.query_linear(query), batch_size)
        key   = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        # Tính attention scores: [batch, n_heads, query_len, key_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            # mask cần có shape [batch, n_heads, query_len, key_len] hoặc broadcast được về nó
            attention_scores += mask  
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        # context: [batch, n_heads, query_len, depth] -> chuyển về [batch, query_len, d_model]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x = torch.tensor(x, dtype=torch.long)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding có thể là tham số học hoặc được tính toán theo công thức; ở đây dùng Parameter cho đơn giản
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # x: [batch, seq_len]
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        logits = self.output_linear(x)
        return logits


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_heads)
        self.attention2 = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # --- Self-Attention cho decoder ---
        # Tạo combined mask cho self-attention của decoder:
        # tgt_mask: [tgt_seq_len, tgt_seq_len]
        # tgt_padding_mask: [batch, tgt_seq_len] → mở rộng thành [batch, 1, 1, tgt_seq_len]
        if tgt_padding_mask is not None:
            expanded_tgt_padding_mask = tgt_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_seq_len]
            padding_mask = expanded_tgt_padding_mask.float() * -1e9  # chuyển các vị trí True thành -inf
            if tgt_mask is not None:
                # Cần mở rộng tgt_mask thành [1, 1, tgt_seq_len, tgt_seq_len] để cộng với padding mask
                expanded_tgt_mask = tgt_mask.unsqueeze(0)  # [1, tgt_seq_len, tgt_seq_len]
                expanded_tgt_mask = expanded_tgt_mask.unsqueeze(1)  # [1, 1, tgt_seq_len, tgt_seq_len]
                combined_tgt_mask = expanded_tgt_mask + padding_mask
            else:
                combined_tgt_mask = padding_mask
        else:
            # Nếu không có padding mask, dùng tgt_mask (cần mở rộng nếu cần)
            combined_tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1) if tgt_mask is not None else None

        attn1_output = self.attention1(x, x, x, mask=combined_tgt_mask)
        x = self.layer_norm1(x + self.dropout(attn1_output))

        # --- Cross-Attention: decoder attend encoder outputs ---
        # Đối với cross attention, ta dùng src_padding_mask:
        if src_padding_mask is not None:
            expanded_src_padding_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_seq_len]
            cross_mask = expanded_src_padding_mask.float() * -1e9
        else:
            cross_mask = None

        attn2_output = self.attention2(x, enc_output, enc_output, mask=cross_mask)
        x = self.layer_norm2(x + self.dropout(attn2_output))

        # --- Feed Forward ---
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(ffn_output))
        return x

class Transformer(torch.nn.Module):
    def __init__(self, en_vocab_size, vi_vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(en_vocab_size, d_model, n_heads, n_layers, d_ff, dropout)
        self.decoder = Decoder(vi_vocab_size, d_model, n_heads, n_layers, d_ff, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        enc_output = self.encoder(src, src_mask)
        logits = self.decoder(tgt, enc_output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        return logits


