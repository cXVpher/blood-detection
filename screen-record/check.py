import torch
print(torch.version.cuda)         
print(torch.cuda.is_available())  

import torchvision
print(torchvision.ops.nms)

import torch
from torchvision.ops import nms

boxes = torch.tensor([[0,0,10,10],[0,0,10,10],[10,10,20,20]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32).cuda()
print(nms(boxes, scores, 0.5))