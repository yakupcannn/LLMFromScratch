import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,d_in,d_out,bias = False):
        super().__init__()
        self.Q = nn.Linear(d_in,d_out,bias = bias)
        self.K = nn.Linear(d_in,d_out,bias=bias)
        self.V = nn.Linear(d_in,d_out,bias = bias)

    def forward(self,X):
        queries = self.Q(X)
        keys = self.K(X)
        values = self.V(X)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5,dim=-1)
        context_vector = attention_weights @ values

        return context_vector
    

class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias = False):
        super().__init__()
        self.Q = nn.Linear(d_in,d_out,qkv_bias)
        self.K = nn.Linear(d_in,d_out,qkv_bias)
        self.V = nn.Linear(d_in,d_out,qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,X):
        num_tokens= X.shape[1]
        queries = self.Q(X)
        keys = self.K(X)
        values = self.V(X)

        attention_scores = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(self.causal_mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        attention_weights = torch.softmax(attention_scores/keys.shape[-1] ** 0.5,dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values

        return context_vector
    


class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in,d_out,context_length,dropout,qkv_bias) for _ in range(n_heads)]
        )
    def forward(self,X):
        return torch.cat([head(X) for head in self.heads],dim = -1)

class MultiHeadAttentionGPT(nn.Module):
    def __init__(self,n_heads,d_in,d_out,context_length,dropout,qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        assert d_out % n_heads == 0,"d_out must be divided by n_heads"

        self.n_heads = n_heads
        self.d_head = d_out // n_heads

        self.Q = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.K = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.V = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out,bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        ##Causal Mask
        # [0,1,1,1,1,1]
        # [0,0,1,1,1,1]
        # [0,0,0,1,1,1] 
        # [0,0,0,0,1,1]
        # [0,0,0,0,0,1]
        # [0,0,0,0,0,0]

        self.register_buffer("causal_mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))


    def forward(self,X):
        b,num_tokens,d_in = X.shape
        queries = self.Q(X) # (b,num_tokens,d_out)
        keys = self.K(X) # (b,num_tokens,d_out)
        values = self.V(X) # (b,num_tokens,d_out)
        
        # (b,num_tokens,d_out) --> (b,num_tokens,n_heads,d_head)
        queries =queries.view((b,num_tokens,self.n_heads,self.d_head))
        keys = keys.view((b,num_tokens,self.n_heads,self.d_head))
        values = values.view((b,num_tokens,self.n_heads,self.d_head))

        # Transpose (b,num_tokens,n_heads,d_head) --> (b,n_heads,num_tokens,d_head)
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # (b,n_heads,num_tokens,d_head) @ (b,n_heads,d_head,num_tokens) --> (b,n_heads,num_tokens,num_tokens)
        attention_scores = queries @ keys.transpose(-2,-1)

        mask_bool = self.causal_mask.bool()[:num_tokens,:num_tokens]

        attention_scores.masked_fill_(mask_bool,-torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5,dim = -1)
        attention_weights = self.dropout(attention_weights)

        #(b,n_heads,num_tokens,num_tokens) @ (b,n_heads,num_tokens,d_head)
        # (b,n_heads,num_tokens,d_head) -> (b,num_tokens,n_heads,d_head)
        context_vecs = (attention_weights @ values).transpose(1,2)
        context_vecs = context_vecs.contiguous().view((b,num_tokens,self.d_out))
        context_vecs = self.out_proj(context_vecs)

        return context_vecs


class LayerNorm(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_size))
        self.shift = nn.Parameter(torch.zeros(embed_size))


    def forward(self,X):
        mean = X.mean(dim = -1, keepdim =True)
        var = X.var(dim= -1,keepdim = True,unbiased = False)
        norm_X = (X - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_X + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.GELU_CONSTANT = 0.044715
    def forward(self,X):
        ##aproximate GELU Func
        return 0.5 * X * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (X + self.GELU_CONSTANT *torch.pow(X,3))))


class FeedForwardBlock(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.seq_layers = nn.Sequential(
            nn.Linear(embed_size,4 * embed_size),
            GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
    def forward(self,X):
        return self.seq_layers(X)
    

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.multihead_att = MultiHeadAttentionGPT(cfg["n_heads"],cfg["embed_size"],cfg["embed_size"],cfg["context_length"],cfg["dropout_rate"],cfg["qkv_bias"])
        self.ff = FeedForwardBlock(cfg["embed_size"])
        self.layernorm1 = LayerNorm(cfg["embed_size"])
        self.layernorm2 = LayerNorm(cfg["embed_size"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self,X):
        #First Layer
        res_X = X
        X = self.layernorm1(X)
        X = self.multihead_att(X)
        X += res_X

        #Second Layer
        res_X = X
        X = self.layernorm2(X)
        X = self.ff(X)
        X += res_X

        return X 


        






## TEST PART ##

X = [[0.43,0.15,0.89], # Your
     [0.55,0.87,0.66], # Journey
     [0.57,0.85,0.64], # Starts
     [0.22,0.58,0.33], # With
     [0.77,0.25,0.10], # One
     [0.05,0.80,0.55]] # Step
X = torch.tensor(X)

#Single Head Attention Test
batch_X = torch.stack((X,X))
b,context_length,d_in = batch_X.shape
if 0:
    causal_attention = CausalAttention(d_in,2,context_length,0.0)
    context_vector = causal_attention(batch_X)
    print(f"Shape:{context_vector.shape}")
##End

# Multihead Attention Test
N_HEADS = 2
DROP_OUT = 0.5
if 0 :
    mulhead_attention = MultiHeadAttention(N_HEADS,d_in,2,context_length,DROP_OUT,qkv_bias=False)
    multi_context_vector = mulhead_attention(batch_X)
    print(multi_context_vector.shape)

##End

# Multihead Attention GPT version Test
if 0:
    gpt_multihead_attention = MultiHeadAttentionGPT(N_HEADS,d_in,6,context_length,DROP_OUT,qkv_bias=False)
    context_vector = gpt_multihead_attention(batch_X)
    print(context_vector.shape)
##End


# Transformer Block Test
if 1:
    CFG = {
        "vocab_size":50527,
        "context_length":1024,
        "embed_size":768,
        "n_heads":12,
        "n_layers":12,
        "dropout_rate":0.1,
        "qkv_bias":False
    }
    X = torch.randn((2,6,768))
    transformer = TransformerBlock(CFG)
    print(transformer(X).shape)









    





