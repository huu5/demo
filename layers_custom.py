import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F


class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=True,
                 res=False):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.self_weight is not None:
        #     stdv = 1. / math.sqrt(self.self_weight.size(1))
        #     self.self_weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        glorot(self.weight)
        if self.self_weight is not None:
            glorot(self.weight)
        zeros(self.bias)

    # 图卷积层GCN的计算公式如下：
    # \[H ^ {(l + 1)} = \sigma(\tilde
    # {D} ^ {-1 / 2} \tilde
    # {A} \tilde
    # {D} ^ {-1 / 2}
    # H ^ {(l)}
    # W ^ {(l)}) \]
    def forward(self, input, adj):  # adj 应为 D^-0.5 * (A+I) * D^-0.5
        support = torch.mm(input, self.weight)  # 形状为 (num_nodes, out_features) 的支持矩阵 support。
        output = torch.spmm(adj, support)  # 进行稀疏矩阵与密集矩阵的乘法，将邻接矩阵 adj 与支持矩阵 support 相乘，
        # 得到新的节点特征矩阵 output，形状为 (num_nodes, out_features)。

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)

        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphBaseBlock(Module):
    """
    The base block for Multi-layer GCN / ResGCN / Dense GCN
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(GraphBaseBlock, self).__init__()
        self.in_features = in_features
        self.hiddendim = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat" and dense == False:
            self.out_features = in_features + out_features
        elif self.aggrmethod == "concat" and dense == True:
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in add model.")
            self.out_features = out_features
        elif self.aggrmethod == "nores":  # no res 即不要残差连接
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat','add' and 'nores'.")

    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                           self.withloop)
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, x)
            x = gc(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)  # 如果 dense 为 False，将最后一层卷积的输出与输入特征矩阵拼接或相加。
        return self._doconcat(x, denseout)  # 如果 dense 为 True，将最后一层卷积的输出与所有层的聚合特征拼接或相加。

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nhiddenlayer,
                                              self.out_features)


class MultiLayerGCNBlock(Module):
    """
    Muti-Layer GCN with same hidden dimension.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="nores", dense=None):
        """
        The multiple layer GCN block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(MultiLayerGCNBlock, self).__init__()

        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=False,
                                    aggrmethod="nores")  # nores 可能表示 "no residual" 或者 "no residual connection"，
        # 这意味着在这个聚合方法中，不使用残差连接。
        self.model.in_features = in_features
        self.model.hiddendim = out_features
        self.model.nhiddenlayer = nbaselayer
        self.model.out_features = out_features
        self.aggrmethod = aggrmethod

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)


# 通过这种方式初始化权重，可以有效地防止深度神经网络训练过程中的梯度消失或爆炸问题，从而提高训练的稳定性和收敛速度。
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
