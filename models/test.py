import torch

r = torch.randn(3, 6)
r = torch.linalg.qr(r)[0]

print(r.shape)