import torch
import torch.backends.cudnn as cudnn
import time
# detector specific imports
from yolact import Yolact 
from data import cfg, set_cfg, set_dataset
from utils.functions import SavePath

img_sizes = [(640, 360), (1280, 720)] 
batch_sizes = [1,2,4,8,16]
test_num = 100
model = 'weights/yolact_plus_base_54_800000.pth' #'weights/yolact_base_54_800000.pth''weights/yolact_plus_base_54_800000.pth'

# set model
model_path = SavePath.from_str(model)
cfg = model_path.model_name + '_config'
set_cfg(cfg)

# set cudnn
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


net = Yolact()
net.load_weights(model)
net.cuda()


print('Model: {}'.format(model))
print('Test number: {}'.format(test_num))

for img_size in img_sizes:
    print()
    print('Input size: {}x{}'.format(img_size[0], img_size[1]))
    
    for batch_size in batch_sizes:
        print('\tBatch size: {}'.format(batch_size))
        x = torch.zeros((batch_size, 3, img_size[0], img_size[1]))
        x.cuda()
        cost = 0
        
        for i in range(test_num):
            torch.cuda.synchronize()
            t0 = time.time()
            y = net(x)
            torch.cuda.synchronize()
            t1 = time.time()
            cost += t1 - t0 # seconds

        cost = cost / test_num
        fps = (test_num * batch_size) / cost

        print('\tAverage spend time {:.2f}ms, {:.2f} sec \n\tfps: {}'.format(cost*1000, cost, fps))
