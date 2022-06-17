import numpy as np
import emo_utils
import emoji
import matplotlib.pyplot as plt

X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')

maxLen = len(max(X_train, key=len).split())
'''
index = 5
print(X_train[index], emo_utils.label_to_emoji(Y_train[index]))
'''

word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')


def sentence_to_avg(sentence, word_to_vec_map):
    """
    将句子转换为单词列表，提取其GloVe向量，然后将其平均。

    参数：
        sentence -- 字符串类型，从X中获取的样本。
        word_to_vec_map -- 字典类型，单词映射到50维的向量的字典

    返回：
        avg -- 对句子的均值编码，维度为(50,)
    """

    # 第一步：分割句子，转换为列表。
    words = sentence.lower().split()

    # 初始化均值词向量
    avg = np.zeros(50, )

    # 第二步：对词向量取平均。
    for w in words:
        avg += word_to_vec_map[w]
    avg = np.divide(avg, len(words))

    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    在numpy中训练词向量模型。

    参数：
        X -- 输入的字符串类型的数据，维度为(m, 1)。
        Y -- 对应的标签，0-7的数组，维度为(m, 1)。
        word_to_vec_map -- 字典类型的单词到50维词向量的映射。
        learning_rate -- 学习率.
        num_iterations -- 迭代次数。

    返回：
        pred -- 预测的向量，维度为(m, 1)。
        W -- 权重参数，维度为(n_y, n_h)。
        b -- 偏置参数，维度为(n_y,)
    """
    np.random.seed(1)

    # 定义训练数量
    m = Y.shape[0]
    n_y = 5
    n_h = 50

    # 使用Xavier初始化参数
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # 将Y转换成独热编码
    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)

    # 优化循环
    for t in range(num_iterations):
        for i in range(m):
            # 获取第i个训练样本的均值
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # 前向传播
            z = np.dot(W, avg) + b
            a = emo_utils.softmax(z)

            # 计算第i个训练的损失
            cost = -np.sum(Y_oh[i] * np.log(a))

            # 计算梯度
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # 更新参数
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t, cost=cost))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b

'''
pred, W, b = model(X_train, Y_train, word_to_vec_map)
print("=====训练集====")
pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
print("=====测试集====")
pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
'''


