from torch import nn
from layers_custom import *
from torch.nn.parameter import Parameter
from utils import sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp


device = torch.device("cuda:0")


class SimPGCN(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False,
                 **kwargs):
        """
        Initial function.
        :param nfeat: the input feature dimension.输入维度
        :param nhid:  the hidden feature dimension.隐藏层维度
        :param nclass: the output feature dimension.分类别数
        :param nhidlayer: the number of hidden blocks.隐藏层数
        :param dropout:  the dropout ratio.随机丢弃
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".中间模块的类型
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".输入层类别
        :param outputlayer: the input layer type, can be "gcn", "dense".输出层类别
        :param nbaselayer: the number of layers in one hidden block.一个模块包含几个隐藏层
        :param activation: the activation function, default is ReLu.什么激活函数
        :param withbn: using batch normalization in graph convolution.批归一化
        :param withloop: using self feature modeling in graph convolution.节点自循环
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".聚合函数
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(SimPGCN, self).__init__()
        # self.mixmode = mixmode
        self.dropout = dropout

        print('=== Number of Total Layers is %s ===' % (nhidlayer * nbaselayer + 2))
        self.BASEBLOCK = MultiLayerGCNBlock

        # input gc
        self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
        baseblockinput = nhid

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()

        self.scores = nn.ParameterList()  # 平衡原始图和特征图效果的分数向量  （n, 1）
        self.scores.append(Parameter(torch.FloatTensor(nfeat, 1)))  # 计算分数向量的权重参数
        for i in range(nhidlayer):
            self.scores.append(Parameter(torch.FloatTensor(nhid, 1)))

        for s in self.scores:
            # s.data.fill_(0)
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
            # glorot(s)
            # zeros(self.bias)

        self.bias = nn.ParameterList()
        self.bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(nhidlayer):
            self.bias.append(Parameter(torch.FloatTensor(1)))
        for b in self.bias:
            b.data.fill_(
                kwargs['bias_init'])  # fill in b with postive value to make score s closer to 1 at the beginning

        # self.D_k = Parameter(torch.FloatTensor(kwargs['nnodes'], 1))
        self.D_k = nn.ParameterList()  # self-loop自适应系数
        self.D_k.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(nhidlayer):
            self.D_k.append(Parameter(torch.FloatTensor(nhid, 1)))
        for Dk in self.D_k:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)
            # glorot(Dk)
        self.identity = sparse_mx_to_torch_sparse_tensor(sp.eye(kwargs['nnodes'])).to(device)

        self.D_bias = nn.ParameterList()
        self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(nhidlayer):
            self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for b in self.D_bias:
            b.data.fill_(0)  # fill in b with postive value to make score s closer to 1 at the beginning

        self.gamma = kwargs['gamma']


    def reset_parameters(self):
        pass

    def forward(self, fea, adj, adj_knn):
        x, _ = self.myforward(fea, adj, adj_knn)
        return x

    def myforward(self, fea, adj, adj_knn, layer=1.5):
        '''output embedding and log_softmax'''
        gamma = self.gamma  # Eq(11)中的超参数



        use_Dk = True
        s_i = torch.sigmoid(fea @ self.scores[0] + self.bias[0])  # Eq(10)

        if use_Dk:
            Dk_i = (fea @ self.D_k[0] + self.D_bias[0])  # Eq(12)
            x = (s_i * self.ingc(fea, adj) + (1-s_i) * self.ingc(fea, adj_knn)) + (gamma) * Dk_i * self.ingc(fea, self.identity)    # Eq(9) + Eq(11)
        else:
            x = s_i * self.ingc(fea, adj) + (1-s_i) * self.ingc(fea, adj_knn)  # Eq(9)

        if layer == 1:
            embedding = x.clone()

        x = F.dropout(x, self.dropout, training=self.training)
        if layer == 1.5:
            embedding = x.clone()

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        if layer == -2:
            embedding = x.clone()

        # output, no relu and dropput here.
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        if use_Dk:
            Dk_o = (x @ self.D_k[-1] + self.D_bias[-1])
            x = (s_o * self.outgc(x, adj) + (1-s_o) * self.outgc(x, adj_knn)) + (gamma) * Dk_o * self.outgc(x, self.identity)
        else:
            x = s_o * self.outgc(x, adj) + (1-s_o) * self.outgc(x, adj_knn)

        if layer == -1:
            embedding = x.clone()
        log_probs = F.log_softmax(x, dim=1)

        self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)
        return log_probs, embedding