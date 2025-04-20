# Kuş uçmaz kervan geçmez bir yerdesin
# Su olsan kimse içmez seni 
# Yol olsan kimse geçmez  
# Elin adamı ne anlar senden
# Çıkarsın bir dağ başına
# Bir agaç bulursun 
# Tellersin pullarsın gelin eylersin
# Bir de bulutları görürsün
# Köpürmüş gelen bulutları
# Başka ne gelir elden
# Çın çın ötüyor yüreğimin kökünde şu dunyanın ıssızlığı
# Tanrı kimseye vermesin böyle bir yanlızlığı 

import torch
from DataClass import DataC
from EqModel import EqModel
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torch.nn import functional as F

dataC = DataC()

device = "cuda"

# LATITUDE ESTIMATION FOR NEXT EARTHQUAKE
col = 2                         # column to be predicted 2:lat, 3:lon, 4:mag, 6:time_diff
p_max = 180                     # latitude max value

B = 2                           # B --> batch size , how many independent sequence will be procssed in parallel
T = 1024                        # T --> 10 lines to generate 201
C = 1190                        # C --> vector size to present any variable 6 element * 16 each one

F = 7                           # number of features per earthquake

lr = 3e-4                       # learning rate/ non scheduled 
dropout = 0.01                  # dropout 

n_embed = C                     # embedding size
n_heads = 8                     # divide m_embed to n_heads pieces (since we have 6 variables)
h_size = n_embed // n_heads     # heads 12 / 4 = 3
n_layer = 8                     # feed forward layer size

max_iters = 4000                 # 
eval_iters = 100

torch.set_float32_matmul_precision('high')

dataC.usgs2DB()                 # usgs to db
dataC.db2File(min_mag=3.9)      # database to latest.csv  (m > 4.9)
dataC.closeAll()                # close db connection etc

data = dataC.getData()          # All data from latest.csv 87.192 x 7

xb, yb = dataC.getBatch(B, T, split='train', col=col)
sizes = dataC.getSizes()

print("\n==============================================")
print(f"x size : {xb.shape}   ")
print(f"y size : {yb.shape}   ")
print(f"n_embed C  : {n_embed:,}  ")
print(f"n_heads    : {n_heads:,}  ")
print(f"h_size     : {h_size:,}   ")
print(f"seq_len T  : {T:}         ")
print(f"n_layers   : {n_layer:,}  ")
print(f"dropout    : {dropout:,}  ")
print(f"learning rt: {lr:,}       ")
print("================================================")
print("")
print("################################################")
print(f"Every {T} lines of input (eq) results 1 output")
print(f"input channel is {F} dims (yr, mt, x, y, m, d, t)")
print(f"output channel is 1 dims (x)                    ")
print(f"x input  shape is {xb.shape}")
print(f"y output shape is {yb.shape}")
print("################################################")
# print(f"\n{1/0}")

eqModel = EqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
eqModel.to(device)
eqModel = torch.compile(eqModel)

eqModel.train()             # switch to train mode
# criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(eqModel.parameters(), lr=lr)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
print("\n\n\n................................................")
print("EPOCHS is starting..............................")
print("................................................\n\n\n")

@torch.no_grad()
def estimate_loss():
    out = {}
    eqModel.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataC.getBatch(B, T, split, col)
            X = X.to(device)
            Y = Y.to(device)    
            _, loss = eqModel(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    eqModel.train()
    return out

@torch.no_grad()
def esimtate_loss_v():
    sum = 0
    for i in range(100):
        x_test, y_true = dataC.getLast(1, T, 'train', col=col)   # 1 line to predict
        x_test = x_test.to(device)
        y_true = y_true.to(device)  
        idx_next = eqModel.generate(x_test)
        diff = idx_next.item()-y_true.item()
        sum = sum + abs(diff)
    return sum/100.0



eqModel.train()
for iter in range(max_iters):
    optimizer.zero_grad(set_to_none=True)

    # print(f"Iter {iter}")
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        loss_v = esimtate_loss_v()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} lossV {loss_v:.4f}  ")
    
    # print("...........................................\n\n")
    # print(f"x.shape in training : {x}")
    # print(f"y.shape in training : {y}")

    x, y = dataC.getBatch(B, T, 'train', col)
    x = x.to(device)
    y = y.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = eqModel(x, y)
    # print(f">>>logits : {logits}")
    # print(f">>>loss : {loss}")               
    loss.backward()    
    optimizer.step()            
    # print(f"Epoch {iter+1}/{max_iters}: Loss={loss}")    
# print(eqModel)
# torch.save(eqModel.state_dict(), 'eqModel_x.pth')
print("----->>>>SYSTEM TEST<<<<-------")

sum = 0
for i in range(100):
    x_test, y_true = dataC.getLast(1, T, 'train', col=col)   # 1 line to predict
    x_test = x_test.to(device)
    y_true = y_true.to(device)  

    print(f"x_test.shape : {x_test.shape}")
    print(f"y_true.shape : {y_true.shape}")

    idx_next = eqModel.generate(x_test)
    diff = idx_next.item()-y_true.item()
    sum = sum + abs(diff)
    print(f"Estimated latitude : {idx_next.item()} Actual latitude  {y_true.item()} Diff : {diff}") 


print(f"sum {sum/100:.4f}")