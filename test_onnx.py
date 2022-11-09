from onnxmodel import ONNXModel
from PIL import Image
import numpy as np
from test_image import WriteDepth
net = ONNXModel("kitti-640.onnx")
limg = np.array(Image.open("/home/ljx/Code/200sever/work/sunhao/Lac-GwcNet/images1/L/13_1664369833690648.L.jpg").convert('RGB')).astype("float32")
limg=np.expand_dims(np.resize(limg,(3,400,640)),0)
# limg=np.expand_dims(limg,0)
rimg = np.array(Image.open("/home/ljx/Code/200sever/work/sunhao/Lac-GwcNet/images1/R/13_1664369833690648.R.jpg").convert('RGB')).astype("float32")
rimg = np.expand_dims(np.resize(rimg,(3,400,640)),0)
# rimg = np.expand_dims(rimg,0)
output  = net.forward(limg,rimg)
limg = np.resize(np.squeeze(limg),(400,640,3))
WriteDepth(output,limg,"result/","L/34_1665285574842567.L.jpg",14.2)
print(1)