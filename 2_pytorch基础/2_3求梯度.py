import torch
x = torch.ones(2, 2, requires_grad=True)

y = x + 2



print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()


# a = torch.randn(2, 2) # 􁗌􀥦􀰘􀙭􀓥􁼕􁦊 requires_grad = False
# a = ((a * 3) / (a - 1))
# print(a.requires_grad) # False
# a.requires_grad_(True)
# print(a.requires_grad) # True
# b = (a * a).sum()
# print(b.grad_fn)

out.backward()


out2 = x.sum()
out2.backward()
out3 = x.sum()
x.grad.data.zero_()
out3.backward()