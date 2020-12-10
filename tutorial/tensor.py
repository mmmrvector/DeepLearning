from __future__ import print_function
import torch
import numpy as np
# construct a 5 x 3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# construct a matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# tensor.Size is infact a tuple, so it supports all tuple operations
print(x.size())

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x)
print(x[:,1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
print(x)
print(y)
print(z)

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

x = torch.randn(1)
print(x)
print(x.item())

# CUDA Tensors
if torch.cuda.is_available():
    print("available")
    device = torch.device("cuda") # a CUDA device object
    y = torch.ones_like(x, device=device) # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("CUDA")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))