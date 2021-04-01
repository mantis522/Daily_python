import torch

a = torch.randn(3, 1, 3)
print(a)

b = torch.max(a, dim=1)
print(b.shape)