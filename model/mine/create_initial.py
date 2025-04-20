import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data

# 读取数据
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['user', 'item', 'category', 'action', 'timestamp'])
    df = df[df['action'].isin([1, 2, 3, 4])]  # 过滤掉异常值
    df['action'] = df['action'].astype(int)  # 确保 action 为整数
    return df

# 构建索引
def build_index(df):
    user_ids = sorted(df['user'].unique())
    item_ids = sorted(df['item'].unique())
    categories = sorted(df['category'].unique())
    
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}
    category_index = {cid: i for i, cid in enumerate(categories)}
    
    print("User Index Sample:", list(user_index.items())[:10])  # 调试输出
    print("Item Index Sample:", list(item_index.items())[:10])
    print("Category Index Sample:", list(category_index.items())[:10])
    
    return user_index, item_index, category_index

# 构建邻接矩阵
def build_adjacency_matrices(df, user_index, item_index, category_index):
    num_users = len(user_index)
    num_items = len(item_index)
    num_categories = len(category_index)
    num_actions = 4  # 直接用 1, 2, 3, 4 作为索引
    
    # 初始化稀疏矩阵存储
    user_item_matrices = {a: defaultdict(int) for a in range(num_actions)}
    user_category_matrices = {a: defaultdict(int) for a in range(num_actions)}
    
    for _, row in df.iterrows():
        u, i, c, a = row['user'], row['item'], row['category'], row['action']
        
        if u not in user_index:
            print(f"Unexpected user ID: {u}")
            continue
        if i not in item_index:
            print(f"Unexpected item ID: {i}")
            continue
        if c not in category_index:
            print(f"Unexpected category ID: {c}")
            continue
        
        a_idx = a - 1  # 直接用 1, 2, 3, 4 - 1 作为索引
        u_idx, i_idx, c_idx = user_index[u], item_index[i], category_index[c]
        
        # 记录用户-商品交互
        user_item_matrices[a_idx][(u_idx, i_idx)] += 1
        
        # 记录用户-商品类别交互
        user_category_matrices[a_idx][(u_idx, c_idx)] += 1
    
    # 转换为稀疏矩阵
    user_item_sparse = {a: coo_matrix((list(mat.values()), zip(*mat.keys())), shape=(num_users, num_items)) for a, mat in user_item_matrices.items()}
    user_category_sparse = {a: coo_matrix((list(mat.values()), zip(*mat.keys())), shape=(num_users, num_categories)) for a, mat in user_category_matrices.items()}
    
    return user_item_sparse, user_category_sparse

# 主函数
def main(file_path):
    df = load_data(file_path)
    user_index, item_index, category_index = build_index(df)
    user_item_sparse, user_category_sparse = build_adjacency_matrices(df, user_index, item_index, category_index)
    return user_item_sparse, user_category_sparse

# 示例调用
# file_path = "data.csv"  # 替换为你的数据文件路径
# user_item_matrices, user_category_matrices = main(file_path)
# print(user_item_matrices)
# print(user_category_matrices)




# 示例调用
file_path = "/home/dongzhi/毕设/data/processed_data/train_data.csv"  # 替换为你的数据文件路径
user_item_matrices, user_category_matrices = main(file_path)
print(user_item_matrices)
print(user_category_matrices)


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")



# 假设您已经有了这四个稀疏矩阵
weights = [0.1, 0.2, 0.3, 0.4]

# 加权求和并转换为稀疏格式
weighted_sum = sum(weights[i] * user_category_matrices[i] for i in range(4))
weighted_sum = sp.coo_matrix(weighted_sum)  # 确保结果是稀疏矩阵

# 将稀疏矩阵转换为PyTorch张量
edge_index = torch.tensor(np.vstack((weighted_sum.row, weighted_sum.col)), dtype=torch.long)
edge_weight = torch.tensor(weighted_sum.data, dtype=torch.float)


# 创建图数据
num_users = 12770
num_items = 2369
data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_users + num_items)

# 定义GCNEmbedding类
class GCNEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEmbedding, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.bn1 = BatchNorm(64, track_running_stats=False)
        self.conv2 = GCNConv(64, out_channels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# 初始化节点特征（假设为单位矩阵）
data.x = torch.eye(num_users + num_items, dtype=torch.float)


# 将模型移到 GPU（如果可用）
model = GCNEmbedding(in_channels=num_users + num_items, out_channels=64).to(device)

# 将图数据移到 GPU（如果可用）
data = data.to(device)
# 定义BPRLoss类
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    
def sample_negative_edges(num_samples, num_users, num_items, pos_edges_set):
    neg_samples = []
    
    # 遍历正样本并为每个正样本采样负样本
    for _ in range(num_samples):
        # 随机选择一个正样本
        u, i = list(pos_edges_set)[np.random.randint(0, len(pos_edges_set))]

        # 获取该用户与之交互的商品集合
        interacted_items = {item for (user, item) in pos_edges_set if user == u}

        # 从该用户未与之交互的商品中采样
        all_items = set(range(num_users, num_users + num_items))
        non_interacted_items = all_items - interacted_items
        neg_item = np.random.choice(list(non_interacted_items))  # 随机选择一个未交互的商品
        
        neg_samples.append((u, neg_item))
    
    # 转换为torch张量
    neg_samples = np.array(neg_samples)
    return torch.tensor(neg_samples, dtype=torch.long)  # 保证输出为 torch 张量

# 训练函数
# 修改训练函数，确保数据和张量都在同一设备上
def train_gcn(model, data, num_users, num_items, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    bpr_loss = BPRLoss()

    # 获取所有正样本对
    pos_edges = data.edge_index.T  # 直接使用 torch 张量

    # 使用集合存储正样本，提高查询速度
    pos_edges_set = set(map(tuple, pos_edges.cpu().numpy()))  # 先移动到 CPU 再转换为 set

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data)

        # 采样正负样本
        pos_indices = np.random.choice(len(pos_edges), 1024, replace=False)
        pos_samples = pos_edges[pos_indices]
        user_indices = pos_samples[:, 0]
        pos_item_indices = pos_samples[:, 1]

        # 将索引移到 GPU（如果需要）
        user_indices = torch.tensor(user_indices, dtype=torch.long).to(device)
        pos_item_indices = torch.tensor(pos_item_indices, dtype=torch.long).to(device)

        # 采样负样本
        neg_samples = sample_negative_edges(1024, num_users, num_items, pos_edges_set)
        neg_item_indices = neg_samples[:, 1].to(device)  # 将负样本移到 GPU

        # 计算正负样本对的得分
        pos_scores = (embeddings[user_indices] * embeddings[pos_item_indices]).sum(dim=1)
        neg_scores = (embeddings[user_indices] * embeddings[neg_item_indices]).sum(dim=1)

        # 数值稳定性处理
        pos_scores = torch.clamp(pos_scores, min=-10, max=10)
        neg_scores = torch.clamp(neg_scores, min=-10, max=10)

        # 计算损失
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("训练完成，用户嵌入和商品种类嵌入已生成。")

# 进行训练
train_gcn(model, data, num_users, num_items)
with torch.no_grad():
    embeddings = model(data)

# 将嵌入存储到文件中
torch.save(embeddings, 'embeddings64.pt')

print("嵌入已保存到文件 'embeddings64.pt'")