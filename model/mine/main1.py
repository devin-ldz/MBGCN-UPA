# =========================================================================
# Copyright (C) 2020-2023. The UltraGCN Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This program bundles some third-party utility functions (hit, ndcg, 
# RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) under
# the MIT License.
#
# Copyright (C) 2020 Xiang Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter


def data_param_prepare(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}

    embedding_dim = config.getint('Model', 'embedding_dim')
    params['embedding_dim'] = embedding_dim
    ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
    params['ii_neighbor_num'] = ii_neighbor_num
    model_save_path = config['Model']['model_save_path']
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')
    params['max_epoch'] = max_epoch

    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')
    
    initial_weight = config.getfloat('Model', 'initial_weight')
    params['initial_weight'] = initial_weight

    dataset = config['Training']['dataset']
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']
    gpu = config['Training']['gpu']
    params['gpu'] = gpu
    device = torch.device('cuda:'+ params['gpu'] if torch.cuda.is_available() else "cpu")
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')
    params['early_stop_epoch'] = early_stop_epoch
    w1 = config.getfloat('Training', 'w1')
    w2 = config.getfloat('Training', 'w2')
    w3 = config.getfloat('Training', 'w3')
    w4 = config.getfloat('Training', 'w4')
    params['w1'] = w1
    params['w2'] = w2
    params['w3'] = w3
    params['w4'] = w4
    negative_num = config.getint('Training', 'negative_num')
    negative_weight = config.getfloat('Training', 'negative_weight')
    params['negative_num'] = negative_num
    params['negative_weight'] = negative_weight

    gamma = config.getfloat('Training', 'gamma')
    params['gamma'] = gamma
    lambda_ = config.getfloat('Training', 'lambda')
    params['lambda'] = lambda_
    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
    params['sampling_sift_pos'] = sampling_sift_pos
    
    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk') 
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']

    # dataset processing
    train_data, test_data, train_mat, user_num, item_num, constraint_mat = load_data(train_file_path, test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=5)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=5)

    params['user_num'] = user_num
    params['item_num'] = item_num

    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    # Compute \Omega to extend UltraGCN to the item-item co-occurrence graph
    ii_cons_mat_path = './' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + dataset + '_ii_neighbor_mat'
    
    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)
    
    # 加载预训练嵌入和商品类别映射
    pretrained_embeddings = load_pretrained_embeddings('embeddings.pt')
    item_category_map = load_item_category_mapping('mine_data/ultragcn_data/item_category.txt')
    
    params['pretrained_embeddings'] = pretrained_embeddings
    params['item_category_map'] = item_category_map
    params['pretrained_dim'] = 16  # 预训练嵌入的维度

    return params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()

    
def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
	neg_candidates = np.arange(item_num)

	if sampling_sift_pos:
		neg_items = []
		for u in pos_train_data[0]:
			probs = np.ones(item_num)
			probs[interacted_items[u]] = 0
			probs /= np.sum(probs)

			u_neg_items = np.random.choice(neg_candidates, size = neg_ratio, p = probs, replace = True).reshape(1, -1)
	
			neg_items.append(u_neg_items)

		neg_items = np.concatenate(neg_items, axis = 0) 
	else:
		neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace = True)
	
	neg_items = torch.from_numpy(neg_items)
	
	return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items


class UltraGCN(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']

        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        # 原始可训练嵌入
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        # 预训练嵌入
        self.pretrained_dim = params.get('pretrained_dim', 16)
        self.pretrained_embeddings = params.get('pretrained_embeddings', None)
        self.item_category_map = params.get('item_category_map', {})
        
        # 冻结的预训练嵌入
        if self.pretrained_embeddings is not None:
            print("预处理预训练嵌入...")
            # 确保预训练嵌入的用户部分与模型用户数匹配
            if len(self.pretrained_embeddings) >= self.user_num:
                self.frozen_user_embeds = self.pretrained_embeddings[:self.user_num].detach().clone()
                print(f"成功加载 {self.user_num} 个用户的预训练嵌入")
            else:
                print(f"警告: 预训练嵌入的用户数 ({len(self.pretrained_embeddings)}) 少于模型用户数 ({self.user_num})")
                # 如果预训练嵌入不足，用零填充
                self.frozen_user_embeds = torch.zeros(self.user_num, self.pretrained_dim)
                if len(self.pretrained_embeddings) > 0:
                    user_count = min(len(self.pretrained_embeddings), self.user_num)
                    self.frozen_user_embeds[:user_count] = self.pretrained_embeddings[:user_count].detach().clone()
            
            # 为商品嵌入创建一个映射层，将类别ID映射到预训练的嵌入
            self.frozen_item_category_embeds = self.create_item_category_embeddings()
        else:
            print("没有找到预训练嵌入，使用零向量替代")
            # 如果没有预训练嵌入，则创建零嵌入
            self.frozen_user_embeds = torch.zeros(self.user_num, self.pretrained_dim)
            self.frozen_item_category_embeds = torch.zeros(self.item_num, self.pretrained_dim)
        
        # 将预训练嵌入注册为缓冲区，这样它们就不会在反向传播中更新
        self.register_buffer('frozen_user_embeds_buffer', self.frozen_user_embeds)
        self.register_buffer('frozen_item_category_embeds_buffer', self.frozen_item_category_embeds)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = params['initial_weight']
        self.initial_weights()

    def create_item_category_embeddings(self):
        """为每个商品ID创建对应的类别嵌入"""
        embeddings = torch.zeros(self.item_num, self.pretrained_dim)
        
        # 统计成功映射的商品数量
        mapped_count = 0
        
        # 遍历所有商品ID，获取其对应的类别嵌入
        for internal_id in range(self.item_num):
            # 在这里，我们假设 self.item_category_map 存储的是原始商品ID到类别ID的映射
            # 但我们需要处理的是内部商品ID（从0开始的索引）
            
            # 尝试不同的方式找到这个内部ID对应的类别
            category_id = None
            
            # 1. 直接尝试使用内部ID查找类别（假设内部ID就是原始ID）
            if internal_id in self.item_category_map:
                category_id = self.item_category_map[internal_id]
            
            # 2. 如果找不到，可能内部ID与原始ID有偏移，尝试其他方式...
            # 这里可以添加其他匹配逻辑
            
            if category_id is not None:
                # 预训练嵌入中的位置：用户数 + 类别ID
                pretrained_idx = self.user_num + category_id
                if pretrained_idx < len(self.pretrained_embeddings):
                    embeddings[internal_id] = self.pretrained_embeddings[pretrained_idx].detach().clone()
                    mapped_count += 1
        
        print(f"成功为 {mapped_count}/{self.item_num} 个商品分配了预训练类别嵌入")
        return embeddings

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def get_concat_embeddings(self, users=None, items=None):
        """获取拼接后的嵌入"""
        device = self.get_device()
        
        user_embeds_concat = None
        item_embeds_concat = None
        
        if users is not None:
            # 获取可训练的用户嵌入
            user_embeds = self.user_embeds(users)
            # 获取冻结的预训练用户嵌入
            frozen_user_embeds = self.frozen_user_embeds_buffer[users].to(device)
            # 拼接两种嵌入
            user_embeds_concat = torch.cat([user_embeds, frozen_user_embeds], dim=-1)
            
        if items is not None:
            # 获取可训练的商品嵌入
            item_embeds = self.item_embeds(items)
            # 获取冻结的预训练商品类别嵌入
            
            # 处理不同形状的items
            if len(items.shape) == 1:
                # 单维度的情况
                frozen_item_embeds = self.frozen_item_category_embeds_buffer[items].to(device)
            else:
                # 多维度的情况 (batch_size, neg_samples)
                # 先展平，获取嵌入，再重塑回原来的形状
                original_shape = items.shape
                flattened_items = items.view(-1)
                frozen_item_embeds = self.frozen_item_category_embeds_buffer[flattened_items].to(device)
                frozen_item_embeds = frozen_item_embeds.view(*original_shape, -1)
                item_embeds = item_embeds.view(*original_shape, -1)
            
            # 拼接两种嵌入
            item_embeds_concat = torch.cat([item_embeds, frozen_item_embeds], dim=-1)
            
        return user_embeds_concat, item_embeds_concat

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        
        # 获取拼接后的嵌入
        user_embeds_concat, pos_embeds_concat = self.get_concat_embeddings(users, pos_items)
        _, neg_embeds_concat = self.get_concat_embeddings(items=neg_items)
      
        # 计算正样本得分
        pos_scores = (user_embeds_concat * pos_embeds_concat).sum(dim=-1) # batch_size
        
        # 处理负样本得分
        if len(neg_items.shape) > 1:
            # 如果是多维负样本
            user_embeds_concat = user_embeds_concat.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            neg_scores = (user_embeds_concat * neg_embeds_concat).sum(dim=-1)  # [batch_size, neg_num]
        else:
            # 单维负样本的情况
            user_embeds_concat = user_embeds_concat.unsqueeze(1)
            neg_scores = (user_embeds_concat * neg_embeds_concat).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        
        # 获取邻居商品的ID
        pos_neighbors = self.ii_neighbor_mat[pos_items].to(device)  # [batch_size, num_neighbors]
        
        # 获取用户嵌入
        user_embeds_concat, _ = self.get_concat_embeddings(users)
        user_embeds_concat = user_embeds_concat.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # 创建邻居嵌入 - 由于形状是[batch_size, num_neighbors]，需要特殊处理
        batch_size, num_neighbors = pos_neighbors.shape
        flattened_neighbors = pos_neighbors.view(-1)  # 展平为一维
        
        # 获取每个邻居的嵌入
        neighbor_item_embeds = self.item_embeds(flattened_neighbors)  # [batch_size*num_neighbors, embedding_dim]
        neighbor_frozen_embeds = self.frozen_item_category_embeds_buffer[flattened_neighbors].to(device)  # [batch_size*num_neighbors, pretrained_dim]
        
        # 拼接并重塑回原始形状
        neighbor_embeds_concat = torch.cat([neighbor_item_embeds, neighbor_frozen_embeds], dim=-1)
        neighbor_embeds_concat = neighbor_embeds_concat.view(batch_size, num_neighbors, -1)  # [batch_size, num_neighbors, embedding_dim+pretrained_dim]
        
        sim_scores = self.ii_constraint_mat[pos_items].to(device)  # [batch_size, num_neighbors]
        
        # 计算损失
        loss = -sim_scores * (user_embeds_concat * neighbor_embeds_concat).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        # 只对可训练的参数计算正则化损失
        for parameter in [self.user_embeds.weight, self.item_embeds.weight]:
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def test_foward(self, users):
        device = self.get_device()
        
        # 获取用户拼接嵌入
        user_embeds = self.user_embeds(users)
        frozen_user_embeds = self.frozen_user_embeds_buffer[users].to(device)
        user_embeds_concat = torch.cat([user_embeds, frozen_user_embeds], dim=-1)
        
        # 获取所有物品拼接嵌入 - 为了避免内存问题，分批处理
        all_item_embeds_concat = []
        batch_size = 10000  # 每批处理的物品数量
        num_batches = (self.item_num + batch_size - 1) // batch_size  # 向上取整
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.item_num)
            
            # 获取当前批次的物品ID
            items_batch = torch.arange(start_idx, end_idx).to(device)
            
            # 获取物品嵌入
            item_embeds_batch = self.item_embeds(items_batch)
            frozen_item_embeds_batch = self.frozen_item_category_embeds_buffer[items_batch].to(device)
            item_embeds_concat_batch = torch.cat([item_embeds_batch, frozen_item_embeds_batch], dim=-1)
            
            all_item_embeds_concat.append(item_embeds_concat_batch)
        
        # 合并所有批次的物品嵌入
        all_item_embeds_concat = torch.cat(all_item_embeds_concat, dim=0)
        
        # 计算用户与所有物品的相似度
        return user_embeds_concat.mm(all_item_embeds_concat.t())

    def get_device(self):
        return self.user_embeds.weight.device


########################### TRAINING #####################################

def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params): 
    device = params['device']
    best_epoch, best_recall, best_ndcg = 0, 0, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    if params['enable_tensorboard']:
        writer = SummaryWriter()
    
    print("使用拼接嵌入进行训练，可训练嵌入维度: {}, 预训练嵌入维度: {}".format(
        params['embedding_dim'], params.get('pretrained_dim', 0)))

    for epoch in range(params['max_epoch']):
        model.train() 
        start_time = time.time()

        for batch, x in enumerate(train_loader): # x: tensor:[users, pos_items]
            users, pos_items, neg_items = Sampling(x, params['item_num'], params['negative_num'], interacted_items, params['sampling_sift_pos'])
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss = model(users, pos_items, neg_items)
            if params['enable_tensorboard']:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if params['enable_tensorboard']:
            writer.add_scalar("Loss/train_epoch", loss, epoch)

        need_test = True
        if epoch < 50 and epoch % 5 != 0:
            need_test = False
            
        if need_test:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])
            if params['enable_tensorboard']:
                writer.add_scalar('Results/recall@20', Recall, epoch)
                writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(), F1_score, Precision, Recall, NDCG))

            if Recall > best_recall:
                best_recall, best_ndcg, best_epoch = Recall, NDCG, epoch
                early_stop_count = 0
                torch.save(model.state_dict(), params['model_save_path'])

            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True
        
        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best recall = {}, best ndcg = {}'.format(best_epoch, best_recall, best_ndcg))
            print('The best model is saved at {}'.format(params['model_save_path']))
            break

    writer.flush()

    print('Training end!')


########################### TESTING #####################################

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
	"""
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
	right_pred = r[:, :k].sum(1)
	precis_n = k
	
	recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
	recall_n = np.where(recall_n != 0, recall_n, 1)
	recall = np.sum(right_pred / recall_n)
	precis = np.sum(right_pred) / precis_n
	return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            rating += mask[batch_users]
            
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        
    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


def load_pretrained_embeddings(file_path):
    """加载预训练好的嵌入向量"""
    if not os.path.exists(file_path):
        print(f"警告：预训练嵌入文件 {file_path} 不存在")
        return None
    return torch.load(file_path)


def load_item_category_mapping(file_path):
    """加载商品与类别的映射关系"""
    if not os.path.exists(file_path):
        print(f"警告：商品类别映射文件 {file_path} 不存在")
        return {}
    
    # 读取原始商品ID到类别ID的映射
    original_item_to_category = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item_id, category_id = map(int, line.strip().split())
                original_item_to_category[item_id] = category_id
    
    print(f"加载了 {len(original_item_to_category)} 个原始商品-类别映射")
    
    # 检查映射的完整性
    min_item_id = min(original_item_to_category.keys()) if original_item_to_category else 0
    max_item_id = max(original_item_to_category.keys()) if original_item_to_category else 0
    print(f"商品ID范围: {min_item_id} 到 {max_item_id}")
    
    # 返回原始映射，在UltraGCN类中处理内部ID到原始ID的转换
    return original_item_to_category


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path')
    args = parser.parse_args()

    print('###################### UltraGCN ######################')

    print('Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(args.config_file)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    # 打印预训练嵌入的信息
    if 'pretrained_embeddings' in params and params['pretrained_embeddings'] is not None:
        print(f"成功加载预训练嵌入，维度为: {params['pretrained_dim']}")
        print(f"预训练嵌入矩阵形状: {params['pretrained_embeddings'].shape}")
    else:
        print("警告: 没有找到预训练嵌入")
    
    if 'item_category_map' in params:
        print(f"加载了 {len(params['item_category_map'])} 个商品-类别映射")
    
    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    
    # 检查冻结嵌入的情况
    print(f"用户冻结嵌入维度: {ultragcn.frozen_user_embeds_buffer.shape}")
    print(f"商品类别冻结嵌入维度: {ultragcn.frozen_item_category_embeds_buffer.shape}")
    
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    print('END')