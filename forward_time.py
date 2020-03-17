import torch
from yolact import Yolact
import time

size = [640, 360] # [1280, 720]
batch_size = 1 # 2, 4
num = 1000

net = Yolact()
net.load_weights('weights/yolact_base_54_800000.pth')

x = torch.zeros((batch_size, 3, size[0], size[1])).cuda()

cost = 0
for i in range(num):
    t0 = time.time()
    y = net(x)
    t1 = time.time()
    cost += t1 - t0

print("input size is {}, test {} times, average spend time {:.2f}ms".format(size, num, cost))