import numpy as np
import random
import time
import cllm_utils


# 获取名称
data = open("dinos.txt", "r").read()

# 转化为小写字符
data = data.lower()

# 转化为无序且不重复的元素列表
chars = list(set(data))

# 获取大小信息
data_size, vocab_size = len(data), len(chars)


char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}


def clip(gradients, maxValue):
    """
    使用maxValue来修剪梯度

    参数：
        gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内

    返回：
        gradients -- 修剪后的梯度
    """
    # 获取参数
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def sample(parameters, char_to_is, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样

    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典
        seed -- 随机种子

    返回：
        indices -- 包含采样字符索引的长度为n的列表。
    """

    # 从parameters 中获取参数
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # 步骤1
    ## 创建独热向量x
    x = np.zeros((vocab_size, 1))

    ## 使用0初始化a_prev
    a_prev = np.zeros((n_a, 1))

    # 创建索引的空列表，这是包含要生成的字符的索引的列表。
    indices = []

    # IDX是检测换行符的标志，我们将其初始化为-1。
    idx = -1

    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    # （我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
    counter = 0
    newline_character = char_to_ix["\n"]

    while (idx != newline_character and counter < 50):
        # 步骤2：使用公式1、2、3进行前向传播
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = cllm_utils.softmax(z)

        # 设定随机种子
        np.random.seed(counter + seed)

        # 步骤3：从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        # 添加到索引中
        indices.append(idx)

        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # 更新a_prev为a
        a_prev = a

        # 累加器
        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix["\n"])

    return indices


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    执行训练模型的单步优化。

    参数：
        X -- 整数列表，其中每个整数映射到词汇表中的字符。
        Y -- 整数列表，与X完全相同，但向左移动了一个索引。
        a_prev -- 上一个隐藏状态
        parameters -- 字典，包含了以下参数：
                        Wax -- 权重矩阵乘以输入，维度为(n_a, n_x)
                        Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                        Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                        b -- 偏置，维度为(n_a, 1)
                        by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
        learning_rate -- 模型学习的速率

    返回：
        loss -- 损失函数的值（交叉熵损失）
        gradients -- 字典，包含了以下参数：
                        dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                        dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                        db -- 偏置的梯度，维度为(n_a, 1)
                        dby -- 输出偏置向量的梯度，维度为(n_y, 1)
        a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
    """

    # 前向传播
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

    # 反向传播
    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients, 5)

    # 更新参数
    parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


def model(data, ix_to_char, char_to_ix, num_iterations=3500,
          n_a=50, dino_names=7, vocab_size=27):
    """
    训练模型并生成恐龙名字

    参数：
        data -- 语料库
        ix_to_char -- 索引映射字符字典
        char_to_ix -- 字符映射索引字典
        num_iterations -- 迭代次数
        n_a -- RNN单元数量
        dino_names -- 每次迭代中采样的数量
        vocab_size -- 在文本中的唯一字符的数量

    返回：
        parameters -- 学习后了的参数
    """

    # 从vocab_size中获取n_x、n_y
    n_x, n_y = vocab_size, vocab_size

    # 初始化参数
    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)

    # 初始化损失
    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

    # 构建恐龙名称列表
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)

    # 初始化LSTM隐藏状态
    a_prev = np.zeros((n_a, 1))

    # 循环
    for j in range(num_iterations):
        # 定义一个训练样本
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
        # 选择学习率为0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = cllm_utils.smooth(loss, curr_loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 2000 == 0:
            print("第" + str(j + 1) + "次迭代，损失值为：" + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                cllm_utils.print_sample(sampled_indices, ix_to_char)

                # 为了得到相同的效果，随机种子+1
                seed += 1

            print("\n")
    return parameters


#开始时间
start_time = time.clock()

#开始训练
parameters = model(data, ix_to_char, char_to_ix, num_iterations=10000)

#结束时间
end_time = time.clock()

#计算时差
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")
