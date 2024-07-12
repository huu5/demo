import random
import dgl
import argparse

import torch
from sklearn.model_selection import train_test_split
from dgl import RowFeatNormalizer
from datasets import *
from utils import *
from selfsl import *


class DataHandlerModule():
	def __init__(self, configuration):
		# [STEP-1-1] Set the arguments with the configuration file.
		self.args = argparse.Namespace(**configuration)
		# [STEP-1-2] Set the seed.
		np.random.seed(self.args.seed)
		random.seed(self.args.seed)
		device = torch.device(self.args.cuda_id)
		torch.cuda.set_device(device)
				
		# [STEP-2] Load dataset.
		print(f"Loading and preprocessing the dataset {self.args.data_name}...")
		"""
		- adj_lists: The list of adjacency matrices for every relations.
		- feat_data: The node feature matrix (np.ndarray format).
		- labels: The labels (np.ndarray format).
		"""
		graph = load_data(self.args.data_name, self.args.multi_relation)
		labels = graph.ndata["label"]
		
		# [STEP-3] Split the train/valid/test dataset with stratified sampling.
		if self.args.data_name.startswith('amazon'):
			idx_unlabeled = 2013 if self.args.data_name == 'amazon_new' else 3305
			# As 0-3304 are unlabeled nodes, they are excepted for the train/valid/test process.
			index = list(range(idx_unlabeled, len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[idx_unlabeled:], stratify=labels[idx_unlabeled:], train_size=self.args.train_ratio, random_state=self.args.seed, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=self.args.test_ratio, random_state=self.args.seed, shuffle=True)
		else:
			index = list(range(len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=self.args.train_ratio, random_state=self.args.seed, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=self.args.test_ratio, random_state=self.args.seed, shuffle=True)

		# [STEP-4] Normalize the node feature matrix and add the self-loop for adjacency matrix.
		if self.args.data_name.startswith('amazon'):
			transform = RowFeatNormalizer(subtract_min=True, node_feat_names=['feature'])
			# 这个转换器的作用是对节点特征进行归一化，其中参数 subtract_min=True 表示减去最小值，
			# node_feat_names=['x'] 表示对节点特征中的 "x" 进行归一化。这个步骤可能有助于提高模型的训练效果或稳定性。
			graph = transform(graph)
		graph.ndata["feature"] = torch.FloatTensor(graph.ndata["feature"]).contiguous()  # 调用 contiguous() 方法，
		# 以确保数据的存储方式是连续的。这是因为 PyTorch 要求数据在内存中是连续存储的。

		sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))  # 参数表示每个节点采样邻居的层数。
		valid_loader = dgl.dataloading.DataLoader(graph, idx_valid, sampler, batch_size=self.args.batch_size, shuffle=False, drop_last=False, use_uva=True)
		test_loader = dgl.dataloading.DataLoader(graph, idx_test, sampler, batch_size=self.args.batch_size, shuffle=False, drop_last=False, use_uva=True)
		
		# [STEP-5] Define the instance variable to handle the data. 
		self.dataset = {'features': graph.ndata["feature"], 'labels': labels, 'graph': graph,
				'idx_train': idx_train,'idx_valid': idx_valid,'idx_test': idx_test,
				'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
				'train_loader': None, 'valid_loader': valid_loader, 'test_loader':test_loader,                                
				'idx_total': list(range(len(labels)))}
		print(f"Finished data loading and preprocessing!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name', type=str, default='amazon')
	parser.add_argument('--self_loop', action='store_true')
	parser.add_argument('--multi_relation', action='store_true')
	parser.add_argument('--train_ratio', type=float, default=0.8)
	parser.add_argument('--test_ratio', type=float, default=0.2)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--emb_size', default=[64, 64])
	parser.add_argument('--cuda_id', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=1024)
	args = parser.parse_args()
	args_map = vars(parser.parse_args())
	data_handler = DataHandlerModule(args_map)
	idx_train = data_handler.dataset['idx_train']
	feat = data_handler.dataset['features']
	train = feat[idx_train]
	print(train)
