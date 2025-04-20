import torch 
import torch.nn as nn
from torch.nn import functional as FN

class PositionalEncoding(nn.Module):
    def __init__(self, T, n_embed):         
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(T, n_embed)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embed)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        y = self.pe[:, :x.size(1)]  
        x = x + y
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class CustomEmbedding(nn.Module):
    
    def __init__(self, sizes, n_embed):     
        # n_embed is the number represent dimension of result
        super().__init__()        
        self.yr_size = sizes['yr_size'] + 1
        self.mt_size = sizes['mt_size'] + 1
        self.x_size  = sizes['x_size'] + 1
        self.y_size  = sizes['y_size'] + 1
        self.m_size  = sizes['m_size'] + 1
        self.d_size  = sizes['d_size'] + 1
        self.t_size  = sizes['t_size'] + 1

        self.n_embed = n_embed
        n2_embed = n_embed // 7                                 # 96 / 6 = 16
     
        self.yr_embed = nn.Embedding(self.yr_size, n2_embed)    # 16
        self.mt_embed = nn.Embedding(self.mt_size, n2_embed)    # 16 
        self.x_embed  = nn.Embedding(self.x_size, n2_embed)     # 16
        self.y_embed  = nn.Embedding(self.y_size, n2_embed)     # 16
        self.m_embed  = nn.Embedding(self.m_size, n2_embed)     # 16
        self.d_embed  = nn.Embedding(self.d_size, n2_embed)     # 16
        self.t_embed  = nn.Embedding(self.t_size, n2_embed)      # 96
    
    def forward(self, data, target=None):
        
        # yr_size =  55   # Max 55 years
        # mt_size =  12   # Max 12 months
        # x_size  = 180   # 0 to 180 degrees
        # y_size  = 360   # 0 to 360 degrees
        # m_size  =  91   # mag 0 to 91
        # d_size  =  70   # depth 0 to 70
        # t_size  = 151   # time 0 to 150 minutes
        
        yr_emb = self.yr_embed(data[:,:,0] )
        mn_emb = self.mt_embed(data[:,:,1] )
        x_emb  = self.x_embed( data[:,:,2] )
        y_emb  = self.y_embed( data[:,:,3] )
        m_emb  = self.m_embed( data[:,:,4] )
        d_emb  = self.d_embed( data[:,:,5] )
        t_emb  = self.t_embed( data[:,:,6] )
        
        emb_ = torch.cat((yr_emb, mn_emb, x_emb, y_emb, m_emb, d_emb, t_emb), dim=-1)
        # emb_all = emb_ + t_emb

        
        # emb_all = yr_emb + mn_emb + x_emb + y_emb + m_emb + d_emb #  + t_emb
        # emb_all = x_emb  + m_emb  #  + t_emb
        # print(f"embed shell {emb_all.shape}")
        return emb_
    
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)   
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attentions in parallel """    
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):        
        out = torch.cat([ h(x) for h  in self.heads], dim=-1)        
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    """ one head for self attention """
    """ input (B,T,C) output (B,T,C) """
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)         # B, T, hs, n_head 
        q = self.query(x)       # B, T, hs, n_head
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**0.5            # (B,T,hs) @ (B, hs, T) -> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))  # (B,T,T)
        wei = FN.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)

        v = self.value(x) # B,T,C
        out = wei @ v # (B,T,T) @ (B,T,c) => (B,T,c)
        return out 
    
class Block(nn.Module):
    def __init__(self, n_embed, n_heads, seq_len, dropout):                 
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_embed, n_heads, head_size, seq_len, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)        

    def forward(self, x):               
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EqModel(nn.Module):
    def __init__(self, sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max):        
        super().__init__()

        self.n_layer = n_layer
        self.B = B
        self.T = T
        self.embed = CustomEmbedding(sizes, n_embed) # (B,T,n_embed)
        self.p_embed = PositionalEncoding(T, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads, T, dropout) for _ in range(n_layer)] )
        self.layer_norm =  nn.LayerNorm(n_embed, T)
        self.lm_linear = nn.Linear(n_embed, p_max, bias=False)    # LATITUDE
        self.apply(self._init_weights)

        self.device = device
        # self.criterion = nn.MSELoss()
    
    def _init_weights(self, module):
        std = 0.2
        if isinstance(module, nn.Linear):
            std = (2*self.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):        
        B, T, F = idx.shape      
        x = self.embed(idx)                     # B,T, 7 ==> B,T,96
        x = x + self.p_embed(x)
        x = self.blocks(x);                     # B,T,96 ==> B,T,96         
        x = self.layer_norm(x);                 # B,T,96 ==> B,T,96
        logits = self.lm_linear(x);             # B,T,96 ==> B,T,180
        # print(f"logits.shape : {logits.shape}")
        # print(f"logits : {logits}")
       
        
        if(targets is None):            
            loss = None
        else:
            B, T, C = logits.shape  # (B,T,180)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)            
            lgt = logits.view(-1, logits.size(-1))
            tgt = targets.view(-1)

            # print(f"lgt.shape : {lgt.shape}")
            # print(f"tgt.shape : {tgt.shape}")
            # print(f"lgt : {lgt}")
            # print(f"tgt : {tgt}")
           
            loss = FN.cross_entropy(lgt, tgt, ignore_index=0)
            # print(f"loss : {loss}")
            # print(f"{1/0}")
        return logits, loss    

    @torch.no_grad()
    def generate(self, x_test):

        #print(f"Inference x_test.shape : {x_test.shape}")
        #print(f"Inference x_test : {x_test}")
        
        logits, _ = self(x_test)

        # print(f"Inference logits.shape : {logits.shape}")
        #print(f"Inference logits1 : {logits}")
       
        
        logits = logits[:, -1, :] # becomes (B, C)
        # print(f"Inference logits2 : {logits}")

        # apply softmax to get probabilities
        probs = FN.softmax(logits, dim=-1) # (B, C)
        #print(f"probs : {probs}")

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        #print(f"idx next : {idx_next}")
        # print(f"{1/0}")

        return idx_next
