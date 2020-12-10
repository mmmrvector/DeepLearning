import torch

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# Do a tensor operation
y = x + 2
print(y)

# y was created as a result of an operation, so it has a grad_fn
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
print(a)
print(a*3)
print(a-1)
a = ((a * 3)/(a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.item())
print(b.grad_fn)

# Gradients
print(x)
print(out)
out.backward()
print(x.grad)