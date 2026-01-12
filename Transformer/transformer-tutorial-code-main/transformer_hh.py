import torch
import torch.nn as nn

# 缩放点积注意力 Multi-Head Attention
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)#???
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  #9,9,9

        values = self.values(values)  # (N, value_len, embed_size) 2,9,512
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)    #2,9,8,64
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm  矩阵乘法

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])#2,8,9,9  第q个词对k个词的打分,
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:#2,1,9,9
            energy = energy.masked_fill(mask == 0, float("-1e20"))#2,8,9,9

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3) #/根号d
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out

# attention+ feed forward
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask) #2,9,512   #mask 2,1,1,9??

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query)) #2,9,512
        forward = self.feed_forward(x) #2,9,512
        out = self.dropout(self.norm2(forward + x)) #2,9,512
        return out


# Encoder模块: 6 * TransformerBlock
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) #10,512
        self.position_embedding = nn.Embedding(max_length, embed_size)#100,512

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): #2,9  2,1,1,9
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) #2,9
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))       # 把2,9变成2,9,512
        ) #2,9,512   至此得到了每个词id的嵌入向量(including position information)
        # self.word_embedding(x) 做的事情：把x中的每个位置的整数当作索引值，从 word_embedding取出对应的向量 。难怪nn.Embedding("10",512)
        # self.position_embedding(positions): 同理，每个词可能有100个维度，难怪nn.Embedding("10*10",512)


        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        # 发现encoder有 5个TransformerBlock
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# attention + 6 * TransformerBlock(=attention + feed Forward)
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


# Decoder模块: 同样由多个DecoderBlock构成
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,#100, 词的最大长度 
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size) # [10,512]把 10个词 id变成词向量
        self.position_embedding = nn.Embedding(max_length, embed_size) # 随机生成[100,512]的可学习参数矩阵

        self.layers = nn.ModuleList(    # 里面有6个decoderblock
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)#2,7
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


# 最终模型
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # 过滤掉无效词(判断值是否为0)，并添加两个维度
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # 2,1,1,9
        # (N, 1, 1, src_len)

        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape

        # 掩码矩阵(下三角为 1 上三角为 0), 对第i个词解码时只能看到第0~i个词
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) #2,1,1,9
        trg_mask = self.make_trg_mask(trg) #2,1,1,7
        enc_src = self.encoder(src, src_mask)#2,9,512
        out = self.decoder(trg, enc_src, src_mask, trg_mask)#2,7,10
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 源序列和目标序列
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device) #x.shape=[2,9]
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device) #trg.shape=[2,8]

    # 填充索引
    src_pad_idx = 0
    trg_pad_idx = 0

    # 词表大小 0~9
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape) # out.shape=[2,7,10]



# embedding:把id变成向量。from(2,9) to (2,9,512)
# linear: 变换向量最后1维度