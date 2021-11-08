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

weights='../YOLOModels/waymococoyolov5m50epoch.pt'
#weights='../YOLOModels/yolov5s.pt'
ckpt = torch.load(weights, map_location=device)  # load checkpoint
for key, value in ckpt.items() :
    print(key) #(key, value)#'model'
modelyaml=ckpt['model'].yaml
print(modelyaml)

csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

print("Model's state_dict:")
for param_tensor in csd:
    print(param_tensor, "\t", csd[param_tensor].size())
#torch.save(csd, '../YOLOModels/yolov5sweights.pt')
from models.experimental import attempt_load
#from TorchClassifier.Datasetutil.Visutil import imshow, 

from models.yolo import Model
nc=modelyaml['nc']
anchors=modelyaml['anchors']
model = Model(modelyaml, ch=3, nc=nc).to(device)  # create
#load the updated model weights
model.load_state_dict(csd, strict=False)  # load
print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
model.float().fuse().eval()

# ckpt = torch.load(weights, map_location=device)
# model=ckpt['model'].float().fuse().eval()
# names = model.module.names if hasattr(model, 'module') else model.names  # get class names

# w=weights
# model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
# stride = int(model.stride.max())  # model stride=32
# names = model.module.names if hasattr(model, 'module') else model.names  # get class names

import cv2
imagepath='./tests/images/bus.jpg'
img0 = cv2.imread(imagepath)  # BGR
assert img0 is not None, f'Image Not Found {imagepath}'
# Padded resize
img_size=640
stride=32
from utils.augmentations import letterbox
import numpy as np
img = letterbox(img0, img_size, stride=stride, auto=True)[0] #(1080, 810, 3)->(640, 480, 3)
# Convert
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB =(3, 640, 480)
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device)
half = False
img = img.half() if half else img.float()  # uint8 to fp16/32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim [1, 3, 640, 480]
preds = model(img)
pred=preds[0] #[1, 3, 80, 60, 85]  [1, 18900, 85]

from utils.general import check_img_size, non_max_suppression
imgsz=[640, 640]
imgsz = check_img_size(imgsz, s=stride)  # check image size


# NMS
classes=None # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False# class-agnostic NMS
conf_thres=0.25# confidence threshold
iou_thres=0.45# NMS IOU threshold
max_det=1000# maximum detections per image
pred_post = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

# Process predictions
for i, det in enumerate(pred_post):  # per image
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        print(f"{n} {names[int(c)]}{'s' * (n > 1)}, ") # add to string