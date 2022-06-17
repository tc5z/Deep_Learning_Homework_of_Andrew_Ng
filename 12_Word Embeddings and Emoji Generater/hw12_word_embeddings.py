import numpy as np
import w2v_utils


words, word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    """
    u与v的余弦相似度反映了u与v的相似程度

    参数：
        u -- 维度为(n,)的词向量
        v -- 维度为(n,)的词向量

    返回：
        cosine_similarity -- 由上面公式定义的u和v之间的余弦相似度。
    """
    distance = 0

    # 计算u与v的内积
    dot = np.dot(u, v)

    # 计算u的L2范数
    norm_u = np.sqrt(np.sum(np.power(u, 2)))

    # 计算v的L2范数
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    # 根据公式1计算余弦相似度
    cosine_similarity = np.divide(dot, norm_u * norm_v)

    return cosine_similarity


'''
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))
'''


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决“A与B相比就类似于C与____相比一样”之类的问题

    参数：
        word_a -- 一个字符串类型的词
        word_b -- 一个字符串类型的词
        word_c -- 一个字符串类型的词
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        best_word -- 满足(v_b - v_a) 最接近 (v_best_word - v_c) 的词
    """

    # 把单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # 获取对应单词的词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    # 获取全部的单词
    words = word_to_vec_map.keys()

    # 将max_cosine_sim初始化为一个比较大的负数
    max_cosine_sim = -100
    best_word = None

    # 遍历整个数据集
    for word in words:
        # 要避免匹配到输入的数据
        if word in [word_a, word_b, word_c]:
            continue
        # 计算余弦相似度
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word


'''
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print('{} -> {} <====> {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))
'''


g = 0.5 * (word_to_vec_map['woman'] - word_to_vec_map['man'] + word_to_vec_map['girl'] - word_to_vec_map['boy'])
'''
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))
'''


def neutralize(word, g, word_to_vec_map):
    """
    通过将“word”投影到与偏置轴正交的空间上，消除了“word”的偏差。
    该函数确保“word”在性别的子空间中的值为0

    参数：
        word -- 待消除偏差的字符串
        g -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_debiased -- 消除了偏差的向量。
    """

    # 根据word选择对应的词向量
    e = word_to_vec_map[word]

    # 根据公式2计算e_biascomponent
    e_biascomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(g))) * g

    # 根据公式3计算e_debiased
    e_debiased = e - e_biascomponent

    return e_debiased


'''
e = "receptionist"
print("去偏差前{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(word_to_vec_map["receptionist"], g)))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("去偏差后{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(e_debiased, g)))
'''


def equalize(pair, bias_axis, word_to_vec_map):
    """
    通过遵循上图中所描述的均衡方法来消除性别偏差。

    参数：
        pair -- 要消除性别偏差的词组，比如 ("actress", "actor")
        bias_axis -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_1 -- 第一个词的词向量
        e_2 -- 第二个词的词向量
    """
    # 第1步：获取词向量
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # 第2步：计算w1与w2的均值
    mu = (e_w1 + e_w2) / 2.0

    # 第3步：计算mu在偏置轴与正交轴上的投影
    mu_B = np.divide(np.dot(mu, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    mu_orth = mu - mu_B

    # 第4步：使用公式7、8计算e_w1B 与 e_w2B
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis

    # 第5步：根据公式9、10调整e_w1B 与 e_w2B的偏置部分
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w1B - mu_B,
                                                                                          np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w2B - mu_B,
                                                                                          np.abs(e_w2 - mu_orth - mu_B))

    # 第6步： 使e1和e2等于它们修正后的投影之和，从而消除偏差
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2


print("==========均衡校正前==========")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("\n==========均衡校正后==========")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
