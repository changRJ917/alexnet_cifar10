import numpy as np


def affine_forward(x, w, b):
    """
    inputs:
    :param x: A numpy array containing input data, of shape (N, d_1,...,d_k)样本,cifar10的shape=(N, 32,32,3)
    :param w: A numpy array of weight, of shape(D,N)权重
    :param b: A numpy array of bias, of shape(M,)偏置，通过numpy加法可实现维度的扩展x.w+b(此时b扩展为N*M)
    returns a tuple of:
    -out: output, of shape (N, M)
    -cache: (x,w,b)
    """
    out = None  # 初始化
    # 确保x是一个规整的矩阵,x.shape表示x的元组大小，x.shape[0]为行数，[1]为列数
    # x.shape为四维矩阵（N, 32,32,3）--> 两维(N, 3072)，-1指剩余可填充的维度
    reshaped_x = np.reshape(x, (x.shape[0], -1))
    # 感知器输出模型，+b自动扩展维度
    out = reshaped_x.dot(w) + b
    # cache表示将输入(x,w,b)存入元组再输出出去
    cache = (x, w, b)

    return out, cache


def relu_forward(x):
    """
    计算relu层的前向运算
    :param x: 任何维度的输入
    :return:与x相同的维度，输出cache
    """
    out = np.maxium(0, x)  # 取x中每个元素与0作比较
    cache = x  # 缓冲输入进来的x矩阵
    return out, cache


def affine_relu_forward(x, w, b):
    """
    卷积层在relu层后作了一步仿射变换
    :param x:
    :param w:
    :param b:
    :return:
    out：output from relu
    cache: object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)  # 线性模型
    out, relu_cache = relu_forward(a)  # 激活函数
    cache = (fc_cache, relu_cache)  # 缓冲的是元组，(x, w, b, (a))
    return out, cache


def softmax_loss(z, y):
    """
    计算softmax分类器的loss和梯度
    :param z: 输入shape(N, C)，表示对第i个输入的第j类z(i, j)的评分score
    :param y: label
    :return:
    loss：loss值
    dz：梯度值
    """
    probs = np.exp(z - np.max(z, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = z.shape[0]
    # 交叉熵在写code的时候记得np.log()使用
    #loss = -np.sum(np.log(probs[np.arange(N), y]))/N
    loss += 0.5 * self.reg * (np.sum(self.params["w1"]**2 +
                                     np.sum(self.params['w2']**2)))

    dz = probs.copy()
    # 该处 -1，是指通过链式求导推导出来的
    dz[np.arange(N), y] -= 1
    dz /= N
    return loss, dz  # dz是关于loss的梯度，dz越小，表明loss波动稳定


def affine_backward(dout, cache):
    """
    计算一个仿射层的后向传播
    :param dout: shape(N, M)上一层的散度输出，对应于上一节softmax定义的dz矩阵
    :param cache: （x, w, b）的元组输出
    :return:
    -dz 相对于z的梯度，（N, d1, ..., d_k）
    -dw 相对于w的梯度, (D, M)
    -db 相对于b的梯度，（M，） ### db的维度如何确定呢
    ## -----说的是（dz, dw, db）的维度与（z, w, b）相同@@@@@@？？？
    """
    z, w, b = cache
    dz, dw, db = None, None, None  # 初始化需要吗？？
    reshaped_x = np.reshape(z, (z.shape[0], -1))
    dz = np.reshape(dout.dot(w.T), z.shape)
    dw = reshaped_x.T.dot(dout)
    db = np.sum(dout, axis=0)  # 这里db.shape = [1, M]
    return dz, dw, db


def relu_backward(dout, cache):
    """
    计算一个relus层的后向传播
    :param dout: 上一层的散度输出
    :param cache: 输入x
    :return:
    dx: 相对于x的梯度
    """
    dx, x = None, cache
    dx = (x > 0) * dout
    # 与所有x中元素为正的位置处，位置相对于dout矩阵的元素保留，其他都取0
    return dx


def affine_relu_backward(dout, cache):
    """
    计算仿射relu卷积层的后向传播
    :param dout:
    :param cache:
    :return:
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def batchnorm_forward(x, gramma, beta, bn_param):
    """

    :param x: data of shape (N, D)
    (gramma, beta)是新的待优化学习的参数
    :param gramma: scale parameter of shape (D,)
    :param beta: shift parameter of shape (D,)
    :param bn_param: dictionary with the following keys:
    mode: 'train' or 'test'
eps: constant for numeric stability 数值稳定性常数
    momentum:动量,constant for running mean/ variance
    running_mean: array of shape(D,),giving running mean of features
    running_var: array of   shape(D,), giving running variance of features
    :return:
    out:of shape(N, D)
    cache: a tuple of values needed in the backward
    """
    mode = bn_param['mode']
    eps = bn_param['eps']  # 数值变量精度，为了避免分母除数为0的情况所使用的微小正数
    momentum = bn_param.get('momentum', 0.9)  # 默认取值为0.9

    N, D = x.shape
    # running_mean, running_var 移动平均值和移动方法，是对x
    # 的每一列算得平均值和标准差向量，如第一层神经元的BN层，相当于把每张
    # 样本图像对应像素维度看作结构化数据的一个特征（3072个），然后算出
    # 所有样本图片中每个像素特征下的平均值和标准差（根据样本的相关值进行更新的）
    running_mean = bn_param.get('running_mean',
                                np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var',
                               np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))
        # 尺度变换，又叫变换重构，引入参数gamma，beta
        # 使得该层神经元可以学习恢复出原始神经网络在该层所要学习的特征分布
        out = gamma * x_hat + beta

        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        out = (x - running_mean) * gamma / (np.sqrt(running_var + eps)) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


class TwoLayerNet(object):
    def __init__(self
                 , input_dim=3*32*32  # 每张样本图片的数据维度大小
                 , hidden_dim=100   # 隐藏层的神经元个数
                 , num_classes=10  # 样本图片的分类类别个数
                 , weight_scale=1e-3):  # 初始化参数的权重尺度（标准偏差）
        self.params = {}
        self.params['w1'] = weight_scale * np.random.randn(input_dim
                                                           , hidden_dim)
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['w2'] = weight_scale * np.random.randn(hidden_dim
                                                           , num_classes)
        self.params['b2'] = np.zeros((num_classes,))

    def loss(self, X, y):
        loss, grads = 0, {}

        h1_out, h1_cache = affine_relu_forward(X,
                                               self.params['w1'],
                                               self.params['b1'])
        scores, out_cache = affine_forward(h1_out,
                                           self.params['w2'],
                                           self.params['b2'])
        loss, dout = softmax_loss(scores, y)
        # 损失值loss的梯度在输出层和隐藏层的反向传播
        dout, dw2, db2 = affine_backward(dout, out_cache)
        grads['w2'] = dw2, grads['b2'] = db2  # 两行并一行，怎么并

        _, dw1, db1 = affine_relu_backward(dout, h1_cache)
        grads['w1'] = dw1, grads['b1'] = db1
        return loss, grads
