import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import cv2

# 给数据类别放置到列表数据中
CLASS_NAMES = np.array(['Cr','In','Pa','Ps','Rs','Sc'])

# 设置图片大小和批次数
IMG_HEIGHT = 32
IM_WIDTH = 32

# 加载模型
model = load_model("model.h5")

# 处理图片
src = cv2.imread('1.png')
src = cv2.resize(src,(32,32))
src = src.astype("int32")
src = src / 255

# 扩充数据得维度
test_img = tf.expand_dims(src,0)

# 进行预测
preds = model.predict(test_img)
score = preds[0]

print('模型预测得结果为{}，概览是{}'.format(CLASS_NAMES[np.argmax(score)],np.max(score)))

