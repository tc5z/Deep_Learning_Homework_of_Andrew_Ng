import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
import kt_utils
import resnets_utils

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def HappyModel(input_shape):
    """
    实现一个检测笑容的模型

    参数：
        input_shape - 输入的数据的维度
    返回：
        model - 创建的Keras的模型

    """

    # 定义一个tensor的placeholder，维度为input_shape
    X_input = Input(input_shape)

    # 使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3, 3))(X_input)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # 降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # 创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


'''
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))

# 创建一个模型实体
happy_model = HappyModel(X_train.shape[1:])
# 编译模型
happy_model.compile("adam", "binary_crossentropy", metrics=['accuracy'])
# 训练模型
happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
# 评估模型
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)

print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))
'''


def identity_block(X, f, filters, stage, block):
    """
    实现恒等块

    参数：
        X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。

    返回：
        X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)

    """

    # 定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 获取过滤器
    F1, F2, F3 = filters

    # 保存输入数据，将会用于为主路径添加捷径
    X_shortcut = X

    # 主路径的第一部分
    ##卷积层
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer='glorot_uniform')(X)
    ##归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    # 主路径的第二部分
    ##卷积层
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer='glorot_uniform')(X)
    ##归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    # 主路径的第三部分
    ##卷积层
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer='glorot_uniform')(X)
    ##归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    ##没有ReLU激活函数

    # 最后一步：
    ##将捷径与输入加在一起
    X = Add()([X, X_shortcut])
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    实现图5的卷积块

    参数：
        X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        s - 整数，指定要使用的步幅

    返回：
        X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
    """

    # 定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 获取过滤器数量
    F1, F2, F3 = filters

    # 保存输入数据
    X_shortcut = X

    # 主路径
    ##主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
               name=conv_name_base + "2a", kernel_initializer="glorot_uniform")(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    ##主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer="glorot_uniform")(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    ##主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer="glorot_uniform")(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # 捷径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid",
                        name=conv_name_base + "1", kernel_initializer="glorot_uniform")(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    # 最后一步
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    实现ResNet50
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    参数：
        input_shape - 图像数据集的维度
        classes - 整数，分类数

    返回：
        model - Keras框架的模型

    """

    # 定义tensor类型的输入数据
    X_input = Input(input_shape)

    # 0填充
    X = ZeroPadding2D((3, 3))(X_input)

    # stage1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv1",
               kernel_initializer="glorot_uniform")(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

    # stage3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # stage4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # stage5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # 均值池化层
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # 输出层
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc" + str(classes),
              kernel_initializer="glorot_uniform")(X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

model.fit(X_train, Y_train, epochs=30, batch_size=32)

preds = model.evaluate(X_test, Y_test)

print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))


# %%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

my_image1 = "1.png"  # 定义图片名称
fileName1 = "datasets/fingers/" + my_image1  # 图片地址
image1 = mpimg.imread(fileName1)  # 读取图片
plt.imshow(image1)
plt.show()
my_image1 = image1.reshape(1, 64, 64, 3)  # 重构图片
print("my_image.shape = " + str(my_image1.shape))

print("finger number is " + str(np.argmax(model.predict(my_image1))))
