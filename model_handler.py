import os
import time
import random
import argparse
from typing import Tuple
import dgl.sparse as dglsp
import torch
import numpy as np
import torch.nn.functional as F
from model_custom import *
from datasets import *
from data_handler import DataHandlerModule
from result_manager import ResultManager
from selfsl import PairwiseAttrSim, MergedKNNGraph, preprocess_adj_noloop
from utils import test, generate_batch_idx

PYG_DIR_PATH = "./data/pyg"


class ModelHandlerModule():
    def __init__(self, configuration, datahandler: DataHandlerModule):
        self.args = argparse.Namespace(**configuration)
        self.dataset = datahandler.dataset
        self.epochs = self.args.epochs
        self.patience = self.args.patience
        self.result = ResultManager(args=configuration)

        # Set the seeds and CUDA ID.
        self.seed = self.args.seed
        device = torch.device(self.args.cuda_id)
        torch.cuda.set_device(device)

        # Select the model according to the json configuration file.
        self.model = self.select_model()
        self.model.cuda()

    def set_seed(self) -> None:
        """
        Set the seed for reproducibility.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def select_model(self) -> nn.Module:
        """
        Select the model according to the configuration.
        If you have imported additional models, you can use it by adding the bellow codes.

        - If the GNN model is homogeneous, adjacemcy matrix is equal to the homogeneous graph.
        - It is determined in the datahandler.
        """
        torch.cuda.empty_cache()
        # graph = self.dataset['graph']
        feature = self.dataset['features']

        model = SimPGCN(feature.shape[1], nhid=self.args.hidden_dim, nclass=2, nhidlayer=self.args.n_layers,
                        dropout=self.args.dropout, device=self.args.cuda_id, bias_init=self.args.bias_init,
                        gamma=self.args.gamma, nnodes=feature.shape[0])

        return model

    def train(self) -> Tuple[np.array, np.array, float]:
        """
        Train SIMPGCN model on the fraud detection dataset.
        It returns
            - Prediction: Pseudo labels of total nodes (n,).
            - Confidence: Confidence score for the prediction of the model (n,).
            - Test AUC-ROC : AUC-ROC of the test set.
        """

        # [STEP-1] Set the seed for various libraries and CUDA ID.
        self.set_seed()
        torch.cuda.empty_cache()
        device = torch.device(self.args.cuda_id)
        torch.cuda.set_device(device)

        # [STEP-2] Define the node indices of train/valid/test process and labels.
        graph = self.dataset['graph']
        idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
        idx_valid, y_valid = self.dataset['idx_valid'], self.dataset['y_valid']
        idx_test, y_test = self.dataset['idx_test'], self.dataset['y_test']

        # [STEP-3-1] Define the model and loss function.
        model = self.model
        # F.cross_entropy 是一个综合函数，结合了 log_softmax 和 nll_loss 两步操作。它可以直接接受原始的未归一化的 logits 作为输入，
        # 并返回交叉熵损失。使用起来非常方便，因为你不需要手动应用 log_softmax。
        # loss_fn = nn.CrossEntropyLoss()  # 在二分类或多分类问题中，
        # 当模型输出的预测概率分布 ( q ) 使用 softmax 函数计算，且真实标签 ( p ) 用 one-hot 编码表示时，
        # 负对数似然损失和交叉熵损失是等价的。这是因为在这种情况下，我们只关心正确类别的概率 ( q(c) )，而忽略其他类别的概率，
        # 因此损失函数都简化为正确类别的负对数概率。这也是为什么在实际应用中，如神经网络的分类任务，这两个术语经常互换使用。

        # # [STEP-3-2] Define the batch sampler.
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))
        #
        # # [STEP-3-3] Initialize the optimizer with learning rate and weight decay rate.
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
        #                              weight_decay=self.args.weight_decay)


        """# knn邻接矩阵"""
        adj1 = graph.adj_tensors(fmt='coo')
        adj = graph.adj().to_dense().to('cuda')
        # adj = preprocess_adj_noloop(adj)
        features = self.dataset['features'].to('cuda')
        ssl_agent = PairwiseAttrSim(adj, features, idx_train=idx_train, nhid=self.args.hidden_dim,
                                    args=self.args,
                                    device='cuda')
        optimizer = torch.optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)  # 两者的参数合并到一个列表中，这样优化器可以同时更新模型和SSL代理的参数。
        tmp_agent = MergedKNNGraph(adj, features, idx_train=idx_train, nhid=self.args.hidden_dim,
                                   args=self.args,
                                   device='cuda')
        adj_knn = tmp_agent.A_feat
        adj_knn = preprocess_adj_noloop(adj_knn, 'cuda')




        # [STEP-3-4] Initialize the performance evaluation measure for validation.
        auc_best, f1_mac_best, epoch_best = 1e-10, 1e-10, 0

        # [STEP-4] Train the model.
        print("\n", "*" * 20, f" Train the SimP-GCN ", "*" * 20)
        for epoch in range(self.epochs):
            model.train()
            avg_loss = []
            optimizer.zero_grad()
            loss, epoch_time = 0.0, 0.0
            torch.cuda.empty_cache()


            # # [STEP-4-1] Generate batch indices for train loader.
            # batch_idx = generate_batch_idx(idx_train, y_train, self.args.batch_size, self.args.seed)  # 生成标签1:1的索引
            # train_loader = dgl.dataloading.DataLoader(graph, batch_idx, sampler, batch_size=self.args.batch_size,
            #                                           shuffle=False, drop_last=False, use_uva=True)


            start_time = time.time()
            optimizer.zero_grad()
            # print('优化器设备：', optimizer.device)
            # optimizer.device = torch.device(self.args.cuda_id)
            # print('当前设备：', torch.cuda.current_device())
            # for param in self.model.parameters():
            #     print("参数：", param)
            #     print('设备：', param.device)

            log_probs, embeddings = model.myforward(features, adj, adj_knn, layer=1.5)
            loss = F.nll_loss(log_probs[idx_train], y_train.cuda())
            # for batch in train_loader:
            #     # [STEP-4-2] Set the batche nodes.
            #     _, output_nodes, blocks = batch
            #     blocks = [b.to(device) for b in blocks]
            #     adj = blocks[0].graph
            #     output_labels = blocks[-1].dstdata['label'].type(torch.LongTensor).cuda()
            #
            #     # [STEP-4=3] Compute the loss of the model.
            #     """
            #      - If loss is nn.CrossEntropyLoss, the data type of the labels should be LongTensor.
            #      - If loss is nn.BCELoss, the data type of the labels should be FloatTensor.
            #     """
            #     logits = model(blocks)
            #     loss = loss_fn(logits, output_labels.squeeze())
            #
            # [STEP-4-4] Compute the gradient and update the model.
            loss.backward()
            optimizer.step()

            # [STEP-4-5] Clear the remain gradient as zero value.
            optimizer.zero_grad()
            avg_loss.append(loss.item() / features.shape[0])  # Calculate average train loss.

            # [STEP-4-6] Write the train log.
            end_time = time.time()
            epoch_time += end_time - start_time
            line = f'Epoch: {epoch + 1} (Best: {epoch_best}), loss: {np.mean(avg_loss)}, time: {epoch_time}s'
            self.result.write_train_log(line, print_line=True)

            # [STEP-5] Validate the model performance for each validation epoch.
            if (epoch + 1) % self.args.valid_epochs == 0:
                model.eval()
                # [STEP-5-1] Calculate the AUC, Recall, F1-macro, Precision with validation set.

                auc_val, recall_val, f1_mac_val, precision_val = test(model, features, idx_valid, y_valid, adj, adj_knn,
                                                                      self.result, epoch_best, flag="val")

                # [STEP-5-2] If the current model is best, save the model and update the best value.
                gain_auc = (auc_val - auc_best) / auc_best
                gain_f1_mac = (f1_mac_val - f1_mac_best) / f1_mac_best
                if (gain_auc + gain_f1_mac) > 0:
                    auc_best, recall_best, f1_mac_best, precision_best, epoch_best = auc_val, recall_val, f1_mac_val, precision_val, epoch
                    torch.save(model.state_dict(), self.result.model_path)

            # [STEP-6] Test early stopping condition.
            if (epoch - epoch_best) > self.args.patience:
                print("\n", "*" * 20, f"Early stopping at epoch {epoch}", "*" * 20, )
                break
        # [STEP-7] Write the best validation results for model selection.
        self.result.write_val_log(auc_best, recall_best, f1_mac_best, precision_best, epoch_best)

        # [STEP-8] Load the best model with repect to the validation performance.
        print("Restore model from epoch {}".format(epoch_best))
        model.load_state_dict(torch.load(self.result.model_path))

        # [STEP-9] Test the model performance.
        print("\n", "*" * 20, f" Test the SimPGCN ", "*" * 20)
        auc_test, recall_test, f1_mac_test, precision_test = test(model, features, idx_test, y_test, adj, adj_knn,
                                                                  self.result, epoch_best, flag="test")

        return auc_test, f1_mac_test


if __name__ == '__main__':
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
