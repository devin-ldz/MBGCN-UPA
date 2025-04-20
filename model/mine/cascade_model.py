import torch
import torch.nn as nn
import torch.nn.functional as F
from main_at import UltraGCN, data_param_prepare, test, Sampling
import os
import time

class CascadeModel(nn.Module):
    def __init__(self, config_paths):
        super(CascadeModel, self).__init__()
        
        # 初始化四种行为的模型
        self.behaviors = ['view', 'fav', 'cart', 'buy']
        self.models = nn.ModuleDict()
        self.transform_matrices = nn.ModuleDict()
        
        # 加载每种行为的配置和模型
        for i, behavior in enumerate(self.behaviors):
            config_path = os.path.join(config_paths, f'{behavior}_config.ini')
            params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(config_path)
            
            # 创建UltraGCN模型
            self.models[behavior] = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
            
            # 创建线性变换矩阵
            if i < len(self.behaviors) - 1:  # 除了最后一个行为，其他都需要变换矩阵
                self.transform_matrices[f'user_{behavior}'] = nn.Linear(params['embedding_dim'], params['embedding_dim'])
                self.transform_matrices[f'item_{behavior}'] = nn.Linear(params['embedding_dim'], params['embedding_dim'])
    
    def forward(self, behavior, users, pos_items, neg_items):
        """前向传播"""
        return self.models[behavior](users, pos_items, neg_items)
    
    def get_embeddings(self, behavior, users=None, items=None):
        """获取指定行为的嵌入"""
        return self.models[behavior].get_embeddings(users, items)
    
    def transform_embeddings(self, behavior, user_embeds, item_embeds):
        """对嵌入进行线性变换"""
        if behavior in self.transform_matrices:
            user_embeds = self.transform_matrices[f'user_{behavior}'](user_embeds)
            item_embeds = self.transform_matrices[f'item_{behavior}'](item_embeds)
        return user_embeds, item_embeds

def train_cascade_model(config_paths, device):
    """训练级联模型"""
    model = CascadeModel(config_paths)
    model = model.to(device)
    
    # 按顺序训练每种行为
    for i, behavior in enumerate(model.behaviors):
        print(f"\n开始训练 {behavior} 行为...")
        
        # 加载当前行为的配置
        config_path = os.path.join(config_paths, f'{behavior}_config.ini')
        params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(config_path)
        
        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        
        # 训练当前行为
        best_recall = 0
        early_stop_count = 0
        
        for epoch in range(params['max_epoch']):
            model.train()
            total_loss = 0
            
            for batch, x in enumerate(train_loader):
                # 对每个批次进行负采样
                users, pos_items, neg_items = Sampling(x, params['item_num'], 
                                                     params['negative_num'], 
                                                     interacted_items, 
                                                     params['sampling_sift_pos'])
                
                users = users.to(device)
                pos_items = pos_items.to(device)
                neg_items = neg_items.to(device)
                
                # 如果是第一个行为，直接训练
                if i == 0:
                    loss = model(behavior, users, pos_items, neg_items)
                else:
                    # 获取前一个行为的嵌入
                    prev_behavior = model.behaviors[i-1]
                    user_embeds, item_embeds = model.get_embeddings(prev_behavior, users, pos_items)
                    
                    # 对嵌入进行变换
                    user_embeds, item_embeds = model.transform_embeddings(prev_behavior, user_embeds, item_embeds)
                    
                    # 将变换后的嵌入作为当前行为的初始嵌入
                    model.models[behavior].user_embeds.weight.data[users] = user_embeds
                    model.models[behavior].item_embeds.weight.data[pos_items] = item_embeds
                    
                    # 计算损失
                    loss = model(behavior, users, pos_items, neg_items)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 评估
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    F1_score, Precision, Recall, NDCG = test(model.models[behavior], test_loader, 
                                                           test_ground_truth_list, mask, 
                                                           params['topk'], params['user_num'])
                    
                    print(f'Epoch {epoch}: Loss = {total_loss:.4f}, '
                          f'F1 = {F1_score:.4f}, Recall = {Recall:.4f}, NDCG = {NDCG:.4f}')
                    
                    # 早停检查
                    if Recall > best_recall:
                        best_recall = Recall
                        early_stop_count = 0
                        # 保存模型
                        torch.save(model.state_dict(), 
                                 os.path.join(config_paths, f'{behavior}_model.pth'))
                    else:
                        early_stop_count += 1
                        if early_stop_count >= params['early_stop_epoch']:
                            print(f'Early stopping at epoch {epoch}')
                            break
        
        print(f"{behavior} 行为训练完成，最佳Recall: {best_recall:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_paths = '/home/dongzhi/毕设/model/mine/mine_data/config'
    train_cascade_model(config_paths, device) 