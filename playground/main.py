import numpy
import torch

prediction = torch.tensor([0.2,0.6,0.1,0.5],dtype=torch.float)
print(prediction)
prediction.apply_(lambda x: 1 if x >= 0.5 else 0)
print(prediction)
correct_tensor = torch.tensor([0,1,1,0])
print(correct_tensor)
output = correct_tensor
torch.logical_xor(correct_tensor,prediction, out=output)
print(output)
torch.logical_not(output,out=output)
print(output)
accuracy = torch.sum(output)/output.size(dim=0)
print(accuracy)
test_tensor = torch.tensor(prediction)
print("correct_tensor: ")
print(correct_tensor)

print(prediction.shape)
print(type(prediction))

list_1 = [i for i in range(5)]
list_2 = [-i for i in range(5)]
list_3 = list(zip(list_1,list_2))
print(list_3)

res = [sum(x) for x in zip(*list_3)]
print(res)

a = numpy.array([1, 2, 3])
t = torch.from_numpy(a)
print(t)