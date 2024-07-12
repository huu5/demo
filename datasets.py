import pandas as pd
import torch
import dgl
from dgl.data.fraud import FraudAmazonDataset, FraudYelpDataset
from sklearn.model_selection import train_test_split

DATA_NAMES = ['yelp', 'amazon', 'amazon_new', 'reddit', 'tfinance', 'tsocial']


def load_data(data_name, multi_relation, self_loop=True, raw_dir='./data', seed=42, train_size=0.1, test_size=0.67):
	
	assert data_name in DATA_NAMES

	if data_name == 'yelp':
		graph = FraudYelpDataset(raw_dir).graph
	elif data_name.startswith('amazon'):
		graph = FraudAmazonDataset(raw_dir).graph
		if data_name == 'amazon_new':
			features = graph.ndata['feature'].numpy()
			mask_dup = torch.BoolTensor(pd.DataFrame(features).duplicated(keep=False).values)
			graph = graph.subgraph(~mask_dup)
	
	# Rename ndata & Remove redundant data
	# graph.ndata['x'] = graph.ndata['feature']
	# graph.ndata['y'] = graph.ndata['label']
	# del graph.ndata['feature'], graph.ndata['label']
	# del graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask']
	if not multi_relation:
		graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'], store_type=False)
		# graph.ndata['x'] = graph.ndata['feature']
		# graph.ndata['y'] = graph.ndata['label']
		# del graph.ndata['_ID'], graph.edata['_ID']
	if self_loop:
		for etype in graph.etypes:
			graph = dgl.add_self_loop(graph, etype=etype)
	labels = graph.ndata["label"]
	# [STEP-3] Split the train/valid/test dataset with stratified sampling.
	# if data_name.startswith('amazon'):
		# idx_unlabeled = 2013 if data_name == 'amazon_new' else 3305
		# As 0-3304 are unlabeled nodes, they are excepted for the train/valid/test process.
		# index = list(range(idx_unlabeled, len(labels)))
		# idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[idx_unlabeled:],
		# 														stratify=labels[idx_unlabeled:],
		# 														train_size=train_size,
		# 														random_state=seed, shuffle=True)
		# idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
		# 														test_size=test_size,
		# 														random_state=seed, shuffle=True)
	# else:
	# 	index = list(range(len(labels)))
		# idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels,
		# 														train_size=train_size,
		# 														random_state=seed, shuffle=True)
		# idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
		# 														test_size=test_size,
		# 														random_state=seed, shuffle=True)

	return graph


if __name__ == '__main__':
	g = load_data('yelp', multi_relation=True)
	print(g.ndata.items())
	# node = g.ndata['_ID']
	# edge = g.edata['_ID']
	train = g.ndata['train_mask']
	print(g.ndata['label'].unique())


