import torch
print(torch.__version__) #1.8.2+cu111
import torchvision
print(torchvision.__version__) #0.9.2+cu111
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

import myyolomulti
from DatasetTools.cocojsontoyolo import make_dirs

from utils.torch_utils import select_device
devicelist='0'
batch_size=16
device = select_device(devicelist,batch_size=batch_size)
print(device)

#weights='../YOLOModels/waymococoyolov5m50epoch.pt'
weights='../YOLOModels/yolov5s.pt'
ckpt = torch.load(weights, map_location=device)  # load checkpoint
for key, value in ckpt.items() :
    print(key) #(key, value)#'model'
modelyaml=ckpt['model'].yaml
print(modelyaml)

csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

print("Model's state_dict:")
for param_tensor in csd:
    print(param_tensor, "\t", csd[param_tensor].size())
torch.save(csd, '../YOLOModels/yolov5sweights.pt')
from models.experimental import attempt_load
#from TorchClassifier.Datasetutil.Visutil import imshow, 