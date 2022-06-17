import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import adam_v2
from td_utils import *
import tensorflow as tf
tf.compat.v1.disable_v2_behavior() # model trained in tf1

Tx = 5511 # 从频谱图输入到模型的时间步数
n_freq = 101 # 在频谱图的每个时间步输入模型的频率数
Ty = 1375 # 我们模型输出中的时间步数

# 使用pydub加载音频片段
activates, negatives, backgrounds = load_raw_audio()


def get_random_time_segment(segment_ms):
    """
    获取 10,000 ms音频剪辑中时间长为 segment_ms 的随机时间段。

    参数：
    segment_ms -- 音频片段的持续时间，以毫秒为单位("ms" 代表 "毫秒")

    返回：
    segment_time -- 以ms为单位的元组（segment_start，segment_end）
    """

    segment_start = np.random.randint(low=0, high=10000 - segment_ms)  # 确保段不会超过10秒背景
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


def is_overlapping(segment_time, previous_segments):
    """
    检查段的时间是否与现有段的时间重叠。

    参数：
    segment_time -- 新段的元组（segment_start，segment_end）
    previous_segments -- 现有段的元组列表（segment_start，segment_end）

    返回：
    如果时间段与任何现有段重叠，则为True，否则为False
    """

    segment_start, segment_end = segment_time

    # 第一步：将重叠标识 overlap 初始化为“False”标志 (≈ 1 line)
    overlap = False

    # 第二步：循环遍历 previous_segments 的开始和结束时间。
    # 比较开始/结束时间，如果存在重叠，则将标志 overlap 设置为True (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    在随机时间步骤中在背景噪声上插入新的音频片段，确保音频片段与现有片段不重叠。

    参数：
    background -- 10秒背景录音。
    audio_clip -- 要插入/叠加的音频剪辑。
    previous_segments -- 已放置的音频片段的时间

    返回：
    new_background -- 更新的背景音频
    """

    # 以ms为单位获取音频片段的持续时间
    segment_ms = len(audio_clip)

    # 第一步：使用其中一个辅助函数来选择要插入的随机时间段
    # 新的音频剪辑。 (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # 第二步：检查新的segment_time是否与previous_segments之一重叠。
    # 如果重叠如果是这样，请继续随机选择新的 segment_time 直到它不重叠。(≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # 第三步： 将新的 segment_time 添加到 previous_segments 列表中 (≈ 1 line)
    previous_segments.append(segment_time)

    # 第四步： 叠加音频片段和背景
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    更新标签向量y。段结尾的后面50个输出的标签应设为 1。
    严格来说，我们的意思是 segment_end_y 的标签应该是 0，而随后的50个标签应该是1。

    参数：
    y -- numpy数组的维度 (1, Ty), 训练样例的标签
    segment_end_ms -- 以ms为单位的段的结束时间

    返回：
    y -- 更新标签
    """

    # 背景持续时间（以频谱图时间步长表示）
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # 将1添加到背景标签（y）中的正确索引
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1

    return y


def create_training_example(background, activates, negatives):
    """
    创建具有给定背景，正例和负例的训练示例。

    参数：
    background -- 10秒背景录音
    activates --  "activate" 一词的音频片段列表
    negatives -- 不是 "activate" 一词的音频片段列表

    返回：
    x -- 训练样例的频谱图
    y -- 频谱图的每个时间步的标签
    """

    # 设置随机种子
    np.random.seed(18)

    # 让背景更安静
    background = background - 20

    # 第一步：初始化 y （标签向量）为0 (≈ 1 line)
    y = np.zeros((1, Ty))

    # 第二步：将段时间初始化为空列表 (≈ 1 line)
    previous_segments = []

    # 从整个 "activate" 录音列表中选择0-4随机 "activate" 音频片段
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # 第三步： 循环随机选择 "activate" 剪辑插入背景
    for random_activate in random_activates:
        # 插入音频剪辑到背景
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # 从 segment_time 中取 segment_start 和 segment_end
        segment_start, segment_end = segment_time
        # 在 "y" 中插入标签
        y = insert_ones(y, segment_end_ms=segment_end)

    # 从整个负例录音列表中随机选择0-2个负例录音
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # 第四步： 循环随机选择负例片段并插入背景中
    for random_negative in random_negatives:
        # 插入音频剪辑到背景
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # 标准化音频剪辑的音量
    background = match_target_amplitude(background, -20.0)

    # 导出新的训练样例
    file_handle = background.export("train" + ".wav", format="wav")
    print("文件 (train.wav) 已保存在您的目录中。")

    # 获取并绘制新录音的频谱图（正例和负例叠加的背景）
    x = graph_spectrogram("train.wav")
    plt.show()

    return x, y


# x, y = create_training_example(backgrounds[0], activates, negatives)


# 加载预处理的训练样例
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
# 加载预处理开发集示例
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


def model(input_shape):
    """
    用 Keras 创建模型的图 Function creating the model's graph in Keras.

    参数：
    input_shape -- 模型输入数据的维度（使用Keras约定）

    返回：
    model -- Keras 模型实例
    """

    X_input = Input(shape=input_shape)

    # 第一步：卷积层 (≈4 lines)
    X = Conv1D(196, 15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization 批量标准化
    X = Activation('relu')(X)  # ReLu activation ReLu 激活
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # 第二步：第一个 GRU 层 (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (使用128个单元并返回序列)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization 批量标准化

    # 第三步: 第二个 GRU 层  (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (使用128个单元并返回序列)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization 批量标准化
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # 第四步： 时间分布全连接层 (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model


'''
model = model((5511, 101))
opt = adam_v2.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, Y, batch_size = 5, epochs=1)
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
'''

model = load_model('./models/tr_model.h5')
loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # 频谱图输出（freqs，Tx），我们想要（Tx，freqs）输入到模型中
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions


chime_file = "audio_examples/chime.wav"


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # 第一步：将连续输出步初始化为0
    consecutive_timesteps = 0
    # 第二步： 循环y中的输出步
    for i in range(Ty):
        # 第三步： 增加连续输出步
        consecutive_timesteps += 1
        # 第四步： 如果预测高于阈值并且已经过了超过75个连续输出步
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # 第五步：使用pydub叠加音频和背景
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # 第六步： 将连续输出步重置为0
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')


#%%
filename = "./raw_data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
