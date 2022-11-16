import time

from onnxmodel import ONNXModel
from PIL import Image
import numpy as np
from test_image import WriteDepthOnnx
from torchvision import transforms

start_time = time.time()

net = ONNXModel("kitti2015-opset11.onnx")

end_time = time.time()
print("load time :",end_time-start_time)

start_time = time.time()
limg_ori = Image.open("images1/L/13_1664369833690648.L.jpg").convert('RGB')
rimg_ori = Image.open("images1/R/13_1664369833690648.R.jpg").convert('RGB')
end_time = time.time()
print("load time :",end_time-start_time)
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



#测试时间时用了for循环，所以表格中不是第一次时间，是跑起来的一个状态
start_time_inter = time.time()
output = net.forward(limg, rimg)
end_time_inter = time.time()
print("interface time :",end_time_inter-start_time_inter)

limg = np.resize(np.squeeze(limg_ori),(400,640,3))
WriteDepthOnnx(output,limg,"result/","L/34_1665285574842567.L.jpg",14.2)
