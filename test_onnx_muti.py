import time

from onnxmodel_feature import ONNXModel as ONNXModelFeature
from onnxmodel_fuse import ONNXModel as ONNXModelFuse
from PIL import Image
import numpy as np
from test_image import WriteDepthOnnx
from torchvision import transforms

load_start_time = time.time()


net_feature = ONNXModelFeature("kitti-feature_extraction.onnx")
net_fuse = ONNXModelFuse("kitti-feature_fuse.onnx")
load_end_time = time.time()
print("load time:", load_end_time-load_start_time)
# limg = np.array(Image.open("/home/ljx/Code/200sever/work/sunhao/Lac-GwcNet/images1/L/13_1664369833690648.L.jpg").convert('RGB')).astype("float32")
# limg=np.expand_dims(np.resize(limg,(3,400,640)),0)
# # limg=np.expand_dims(limg,0)
# rimg = np.array(Image.open("/home/ljx/Code/200sever/work/sunhao/Lac-GwcNet/images1/R/13_1664369833690648.R.jpg").convert('RGB')).astype("float32")
# rimg = np.expand_dims(np.resize(rimg,(3,400,640)),0)
# # rimg = np.expand_dims(rimg,0)

limg_ori = Image.open("images1/L/13_1664369833690648.L.jpg").convert('RGB')
rimg_ori = Image.open("images1/R/13_1664369833690648.R.jpg").convert('RGB')

# why crop
w, h = limg_ori.size
# limg = limg.crop((w - 1232, h - 368, w, h))
# rimg = rimg.crop((w - 1232, h - 368, w, h))

limg_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg_ori)
rimg_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg_ori)
limg_tensor = limg_tensor.unsqueeze(0).cuda()
rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

limg=limg_tensor.cpu().numpy()
rimg=rimg_tensor.cpu().numpy()


output_feature_l  = net_feature.forward(limg)
output_feature_r  = net_feature.forward(rimg)

#测试时间时用了for循环，所以表格中不是第一次时间，是跑起来的一个状态
inter_time_start = time.time()
output=net_fuse.forward(limg,output_feature_l[0],output_feature_r[0])

end_time_start = time.time()
print(end_time_start-inter_time_start)

limg = np.resize(np.squeeze(limg_ori),(400,640,3))
WriteDepthOnnx(output,limg,"result/","L/34_1665285574842567.L.jpg",14.2)
