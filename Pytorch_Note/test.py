import torch as t

x = t.rand(5,3)
print("随机矩阵ｘ")
print(x)

y = t.rand(5,3)
print("随机矩阵ｙ")
print(y)

if t.cuda.is_available():
	x = x.cuda()
	y = y.cuda()
	print(x+y)
