import sys
import os
import json

from collections import defaultdict
import numpy as np

import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import pickle as pkl

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config
		
	def load_dataSet(self, dataSet='cora'):
		if dataSet == 'cora':
			cora_content_file = self.config['file_path.cora_content']
			cora_cite_file = self.config['file_path.cora_cite']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			with open(cora_content_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			feat_data = np.asarray(feat_data)
			# print(feat_data.size())
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(cora_cite_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 2
					paper1 = node_map[info[0]]
					paper2 = node_map[info[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)

			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)

		elif dataSet == 'pubmed':
			pubmed_content_file = self.config['file_path.pubmed_paper']
			pubmed_cite_file = self.config['file_path.pubmed_cites']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(pubmed_content_file) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(pubmed_cite_file) as fp:
				fp.readline()
				fp.readline()
				for line in fp:
					info = line.strip().split("\t")
					paper1 = node_map[info[1].split(":")[1]]
					paper2 = node_map[info[-1].split(":")[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
			
			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)

		elif dataSet == 'reddit':
			reddit_dir = self.config['file_path.reddit_dir']

			# transfer to NPZ data 
			if not os.path.exists(reddit_dir+"reddit.npz"):
			# if 1:
				G = json_graph.node_link_graph(json.load(open(reddit_dir + "/reddit-G.json")))

				# Remove all nodes that do not have val/test annotations
				# (necessary because of networkx weirdness with the Reddit data)
				broken_count = 0
				for node in list(G.nodes()):
					# if not 'val' in G.node[node] or not 'test' in G.node[node]:
					if G.node[node] == {}:
						G.remove_node(node)
						broken_count += 1
				print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

				## Make sure the graph has edge train_removed annotations
				## (some datasets might already have this..)
				print("Loaded data... now preprocessing...")
				for edge in G.edges():
					if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
						G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
						G[edge[0]][edge[1]]['train_removed'] = True
					else:
						G[edge[0]][edge[1]]['train_removed'] = False

				self.transferRedditDataFormat(G, reddit_dir)
				self.transferRedditData2AdjNPZ(G, reddit_dir)
			
			# load from NPZ
			adj_lists, feat_data, labels, train_indexs, val_indexs, test_indexs = self.loadRedditFromNPZ(reddit_dir)
			assert len(feat_data) == len(labels) == len(adj_lists)
			
			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)


	def _split_data(self, num_nodes, test_split = 3, val_split = 6):
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		
		return test_indexs, val_indexs, train_indexs


	def loadRedditFromG(self, dataset_dir, inputfile):
		f= open(dataset_dir+inputfile)
		objects = []
		for _ in range(pkl.load(f)):
			objects.append(pkl.load(f))
		adj, train_labels, val_labels, test_labels, train_index, val_index, test_index = tuple(objects)
		feats = np.load(dataset_dir + "/reddit-feats.npy")
		return sp.csr_matrix(adj), sp.lil_matrix(feats), train_labels, val_labels, test_labels, train_index, val_index, test_index


	def loadRedditFromNPZ(self, dataset_dir):
		adj_data = np.load(dataset_dir+"reddit_adj.npz")
		adj = adj_data['arr_0'].item()
		data = np.load(dataset_dir+"reddit.npz")

		# return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']
		return adj, data['feats'], data['labels'], data['train_index'], data['val_index'], data['test_index']


	def transferRedditData2AdjNPZ(self, G, dataset_dir):
		# G = json_graph.node_link_graph(json.load(open(dataset_dir + "/reddit-G.json")))
		# feat_id_map = json.load(open(dataset_dir + "/reddit-id_map.json"))
		# feat_id_map = {id: val for id, val in feat_id_map.items()}
		labels = json.load(open(dataset_dir + "/reddit-class_map.json"))
		ids = list(labels.keys())
		vals = list(labels.values())

		adj_lists = defaultdict(set)
		# [adj_lists[feat_id_map[id]].add('') for id in ids]
		[adj_lists[id].add('') for id in ids]
		with open(dataset_dir + "/reddit-adjlist.txt") as fp:
			for line in fp:
				if '#' in line:
					continue
				info = line.strip().split()
				# if info[0] in G.nodes():
				for inf in info[1:]:
					# adj_lists[feat_id_map[info[0]]].add(feat_id_map[inf])
					# adj_lists[feat_id_map[inf]].add(feat_id_map[info[0]])
					adj_lists[info[0]].add(inf)
					adj_lists[inf].add(info[0])
		# [adj_lists[feat_id_map[id]].remove('') for id in ids]
		[adj_lists[id].remove('') for id in ids]
		np.savez(dataset_dir + "reddit_adj.npz", adj_lists)

		# adj = dict.fromkeys(list(G.nodes()))
		# numNode = len(adj)
		# print(numNode)
		# for adj_line in open(dataset_dir + "/reddit-adjlist.txt"):
		# 	if '#' in adj_line:
		# 		continue
		# 	adj_mem = set()
		# 	links = adj_line.strip('\n').split(' ')
		# 	if links[0] in G.nodes():
		# 		for link in links[1:]:
		# 			adj_mem.add(link)
		# 		# if links[0] in 
		# 		adj[links[0]] = adj_mem
		


	def transferRedditDataFormat(self, G, dataset_dir):
		# G = json_graph.node_link_graph(json.load(open(dataset_dir + "/reddit-G.json")))
		labels = json.load(open(dataset_dir + "/reddit-class_map.json"))

		train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
		test_ids = [n for n in G.nodes() if G.node[n]['test']]
		val_ids = [n for n in G.nodes() if G.node[n]['val']]
		train_labels = [labels[i] for i in train_ids]
		test_labels = [labels[i] for i in test_ids]
		val_labels = [labels[i] for i in val_ids]
		# all_nodes = labels.keys()
		all_labels = list(labels.values())
		feats = np.load(dataset_dir + "/reddit-feats.npy")

		## Logistic gets thrown off by big counts, so log transform num comments and score
		feats[:, 0] = np.log(feats[:, 0] + 1.0)
		feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
		feat_id_map = json.load(open(dataset_dir + "reddit-id_map.json"))
		feat_id_map = {id: val for id, val in feat_id_map.items()}

		train_index = [feat_id_map[id] for id in train_ids]
		val_index = [feat_id_map[id] for id in val_ids]
		test_index = [feat_id_map[id] for id in test_ids]
		np.savez(dataset_dir + "reddit.npz", feats=feats, labels=all_labels, y_train=train_labels, y_val=val_labels, y_test=test_labels,
				train_index=train_index,
				val_index=val_index, test_index=test_index)

