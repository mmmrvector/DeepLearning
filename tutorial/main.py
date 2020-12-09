from __future__ import print_function
import torch

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