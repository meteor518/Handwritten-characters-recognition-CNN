from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from data import load_data
import random

# 加载数据
data, label = load_data()
# 打乱数据
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
# tets_data
# test_label
print(data.shape[0], ' samples')

# label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 10)

###############  
# 开始建立CNN模型
###############  

# 生成一个model
model = Sequential()

# 第一个卷积层
model.add(Convolution2D(4, 3, 3, activation='relu', input_shape=(1, 28, 28)))

# 第二个卷积层
model.add(Convolution2D(8, 3, 3, activation='relu'))

# 下采样
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第三个卷积层
model.add(Convolution2D(16, 3, 3, activation='relu'))
# 下采样
model.add(MaxPooling2D(pool_size=(2, 2)))

# 全连接层
model.add(Flatten())
model.add(Dense(128, activation='relu', init='normal'))

# Softmax分类，输出是10类别
model.add(Dense(10, activation='softmax', init='normal'))

# 编译
sgd = SGD( lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# 训练
model.fit(data, label, batch_size=100, nb_epoch=10, shuffle=True, verbose=1, show_accuracy=True, validation_split=0.2)

# # 测试
# score = model.evaluate(test_data, test_label)