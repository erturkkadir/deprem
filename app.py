"""
Earthquake Prediction Model Training Script
Uses ComplexEqModel with complex-valued embeddings and attention as specified in CLAUDE.md
"""
import torch
from DataClass import DataC
from EqModel import ComplexEqModel, EqModel
import numpy as np
from torch.nn import functional as F
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train earthquake prediction model')
parser.add_argument('--use-complex', action='store_true', default=True,
                    help='Use complex-valued model (default: True)')
parser.add_argument('--epochs', type=int, default=4000, help='Number of training iterations')
parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
args = parser.parse_args()

dataC = DataC()

device = "cuda" if torch.cuda.is_available() else "cpu"

# LATITUDE ESTIMATION FOR NEXT EARTHQUAKE
col = 2                         # column to be predicted 2:lat, 3:lon, 4:mag, 6:time_diff
p_max = 180                     # latitude max value

B = args.batch_size             # batch size
T = args.seq_len                # sequence length
C = 1190                        # embedding size (7 features * 170 each)

F_FEATURES = 7                  # number of features per earthquake

lr = 3e-4                       # learning rate
dropout = 0.01                  # dropout

n_embed = C                     # embedding size
n_heads = 8                     # attention heads
h_size = n_embed // n_heads     # head size
n_layer = 8                     # transformer blocks

max_iters = args.epochs
eval_iters = 100

data = dataC.getData()          # Load data from latest.csv
xb, yb = dataC.getBatch(B, T, split='train', col=col)
sizes = dataC.getSizes()

print("\n" + "=" * 50)
print("EARTHQUAKE PREDICTION MODEL TRAINING")
print("=" * 50)
print(f"Model Type  : {'ComplexEqModel' if args.use_complex else 'EqModel'}")
print(f"Device      : {device}")
print(f"x size      : {xb.shape}")
print(f"y size      : {yb.shape}")
print(f"n_embed     : {n_embed:,}")
print(f"n_heads     : {n_heads:,}")
print(f"h_size      : {h_size:,}")
print(f"seq_len T   : {T}")
print(f"n_layers    : {n_layer:,}")
print(f"dropout     : {dropout}")
print(f"learning rt : {lr}")
print(f"max_iters   : {max_iters}")
print("=" * 50 + "\n")

# Create model based on argument
if args.use_complex:
    eqModel = ComplexEqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
    model_save_path = 'eqModel_complex.pth'
else:
    eqModel = EqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
    model_save_path = 'eqModel.pth'

eqModel.to(device)
eqModel = torch.compile(eqModel)

eqModel.train()
optimizer = torch.optim.AdamW(eqModel.parameters(), lr=lr)

print("\n" + "." * 50)
print("TRAINING STARTED...")
print("." * 50 + "\n")

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
def estimate_loss_v():
    sum = 0
    for i in range(100):
        x_test, y_true = dataC.getLast(1, T, 'train', col=col)   # 1 line to predict
        x_test = x_test.to(device)
        y_true = y_true.to(device)  
        idx_next = eqModel.generate(x_test)
        diff = idx_next.item()-y_true.item()
        sum = sum + abs(diff)
    return sum/100.0



best_val_loss = float('inf')
eqModel.train()

for iter in range(max_iters):
    optimizer.zero_grad(set_to_none=True)

    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        loss_v = estimate_loss_v()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, avg diff {loss_v:.4f}")

        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(eqModel.state_dict(), model_save_path)
            print(f"  -> Model saved to {model_save_path}")

    x, y = dataC.getBatch(B, T, 'train', col)
    x = x.to(device)
    y = y.to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = eqModel(x, y)

    loss.backward()
    optimizer.step()

# Final save
torch.save(eqModel.state_dict(), model_save_path)
print(f"\nTraining complete. Model saved to {model_save_path}")

print("\n" + "=" * 50)
print("SYSTEM TEST - 100 Predictions")
print("=" * 50)

total_diff = 0
results = []
for i in range(100):
    x_test, y_true = dataC.getLast(1, T, 'train', col=col)
    x_test = x_test.to(device)
    y_true = y_true.to(device)

    idx_next = eqModel.generate(x_test)
    predicted = idx_next.item()
    actual = y_true.item()
    diff = predicted - actual
    total_diff += abs(diff)

    results.append({
        'predicted': predicted - 90,  # Convert to real latitude
        'actual': actual - 90,
        'diff': diff
    })
    print(f"[{i+1:3d}] Predicted: {predicted:3d} ({predicted-90:+4d}) | Actual: {actual:3d} ({actual-90:+4d}) | Diff: {diff:+4d}")

avg_diff = total_diff / 100
print("\n" + "-" * 50)
print(f"Average absolute difference: {avg_diff:.2f} degrees")
print(f"Model accuracy within 5 degrees: {sum(1 for r in results if abs(r['diff']) <= 5)}%")
print(f"Model accuracy within 10 degrees: {sum(1 for r in results if abs(r['diff']) <= 10)}%")
print("-" * 50)