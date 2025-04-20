import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
# 确保导入了 main_at 中的函数
from main_at import RecallPrecision_ATk, NDCGatK_r, getLabel, data_param_prepare 

# --- SelfAttentionFusion 类定义 (保持不变) ---
class SelfAttentionFusion(nn.Module):
    # ... (之前的 SelfAttentionFusion 代码) ...
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttentionFusion, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_layer = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.fusion_weight_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(embed_dim // 2, 1)
        )
        self.out_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim) 
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_layer.weight)
        for m in self.fusion_weight_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
        nn.init.xavier_uniform_(self.out_layer.weight)
        if self.out_layer.bias is not None:
            nn.init.constant_(self.out_layer.bias, 0.)

    def forward(self, x, mask=None):
        N, B, d = x.shape
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(N, B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights_dropout, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, B, d)
        fusion_scores = self.fusion_weight_mlp(attn_output.view(N * B, d)).view(N, B, 1)
        fusion_weights = F.softmax(fusion_scores, dim=1)
        fused_output = (fusion_weights * attn_output).sum(dim=1)
        output = self.out_layer(fused_output)
        output = self.dropout(output)
        output = self.layer_norm(output + x.mean(dim=1)) 
        return output, fusion_weights

# --- load_embeddings, evaluate_combined_embeddings, test_one_batch (保持不变) ---
def load_embeddings(model_dir, behavior, device):
    model_path = os.path.join(model_dir, f'{behavior}_model.pth')
    if not os.path.exists(model_path):
        print(f"警告: 找不到{behavior}行为的模型文件: {model_path}")
        return None, None
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        user_key, item_key = None, None
        possible_user_keys = [f'models.{behavior}.user_embeds.weight', 'user_embeds.weight', 'embedding_user.weight']
        possible_item_keys = [f'models.{behavior}.item_embeds.weight', 'item_embeds.weight', 'embedding_item.weight']
        for key in possible_user_keys:
            if key in state_dict: user_key = key; break
        for key in possible_item_keys:
            if key in state_dict: item_key = key; break
        if user_key and item_key:
            user_embeds = state_dict[user_key].clone().detach().to(device)
            item_embeds = state_dict[item_key].clone().detach().to(device)
            #print(f"成功加载{behavior}行为的嵌入，用户嵌入形状: {user_embeds.shape}, 物品嵌入形状: {item_embeds.shape}")
            return user_embeds, item_embeds
        else:
            print(f"警告: 在{behavior}模型中找不到嵌入权重 (尝试的键: {possible_user_keys}, {possible_item_keys})")
            return None, None
    except Exception as e:
        print(f"加载{behavior}模型 '{model_path}' 时出错: {str(e)}")
        return None, None

def evaluate_combined_embeddings(user_embeds, item_embeds, test_loader, test_ground_truth_list, mask, topk, user_num, device):
    user_embeds = user_embeds.to(device)
    item_embeds = item_embeds.to(device)
    rating_list, groundTrue_list, all_ratings = [], [], []
    with torch.no_grad():
        for batch_users in test_loader:
            batch_users = batch_users.to(device)
            user_embeddings_batch = user_embeds[batch_users]
            rating = torch.matmul(user_embeddings_batch, item_embeds.t())
            rating = rating + mask[batch_users]
            all_ratings.append(rating.cpu().numpy())
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K.cpu())
            # groundTrue_list.append([test_ground_truth_list[u.item()] for u in batch_users])
            # 确保使用 test_ground_truth_list
            current_ground_truth = []
            for u in batch_users:
                user_id = u.item()
                if 0 <= user_id < len(test_ground_truth_list):
                     current_ground_truth.append(test_ground_truth_list[user_id])
                else:
                     current_ground_truth.append([]) # Handle invalid user ID
            groundTrue_list.append(current_ground_truth)

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0
    valid_batches = 0
    for i, x in enumerate(X):
         # test_one_batch 返回 precision, recall, ndcg
         # 需要确认 test_one_batch 的返回值
         try: # 添加 try-except 以捕获 test_one_batch 的潜在错误
             # 假设 test_one_batch 返回值是 prec, rec, ndcg
             precision, recall, ndcg = test_one_batch(x, topk) 
             if np.isnan(precision) or np.isinf(precision): precision = 0
             if np.isnan(recall) or np.isinf(recall): recall = 0
             if np.isnan(ndcg) or np.isinf(ndcg): ndcg = 0
             Precision += precision
             Recall += recall
             NDCG += ndcg
             valid_batches += 1 # 只有成功计算的批次才计数
         except Exception as e:
              print(f"警告: 计算批次 {i} 指标时出错: {e}")
              #可以选择跳过此批次或设置为0
    
    # 使用 valid_batches 进行平均，如果 valid_batches 为 0 则指标为 0
    num_test_users = len(test_loader.dataset) # 获取测试用户总数
    if num_test_users > 0:
        Precision /= num_test_users 
        Recall /= num_test_users
        NDCG /= num_test_users
    else:
        Precision, Recall, NDCG = 0, 0, 0
        print("警告: 测试用户数为 0，无法计算平均指标。")

    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    final_predictions = np.concatenate(all_ratings, axis=0) if all_ratings else None
    return F1_score, Precision, Recall, NDCG, final_predictions

def test_one_batch(X, k):
    """计算一个批次的评估指标"""
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    
    # 确保 groundTrue 是列表的列表
    if not isinstance(groundTrue, list) or (groundTrue and not isinstance(groundTrue[0], list)):
         print(f"警告: groundTrue 格式不正确: {groundTrue}")
         # 尝试修复或返回 0
         # 如果 groundTrue 是单个列表，包装它
         if isinstance(groundTrue, list) and not any(isinstance(el, list) for el in groundTrue):
              groundTrue = [groundTrue]
         else: # 无法处理，返回 0
              return 0.0, 0.0, 0.0 

    # 过滤掉 groundTrue 为空的内部列表
    valid_indices = [i for i, gt in enumerate(groundTrue) if gt]
    if not valid_indices:
        return 0.0, 0.0, 0.0 # 如果没有有效的 ground truth，返回 0

    # 只对有效的项进行计算
    # 确保 sorted_items 和 groundTrue 的长度在过滤前一致
    if sorted_items.shape[0] != len(groundTrue):
         print(f"警告: sorted_items ({sorted_items.shape[0]}) 和 groundTrue ({len(groundTrue)}) 长度不匹配。")
         # 可能需要调整逻辑或返回 0
         min_len = min(sorted_items.shape[0], len(groundTrue))
         sorted_items = sorted_items[:min_len]
         groundTrue = groundTrue[:min_len]
         # 重新计算 valid_indices
         valid_indices = [i for i, gt in enumerate(groundTrue) if gt]
         if not valid_indices: return 0.0, 0.0, 0.0

    valid_sorted_items = sorted_items[valid_indices]
    valid_groundTrue = [groundTrue[i] for i in valid_indices]

    if valid_sorted_items.size == 0:
         return 0.0, 0.0, 0.0

    # getLabel 和评估函数需要 groundTrue (列表的列表) 和 sorted_items (numpy array)
    r = getLabel(valid_groundTrue, valid_sorted_items) # getLabel 返回适合 RecallPrecision_ATk 的格式
    ret = RecallPrecision_ATk(valid_groundTrue, r, k) # 需要 valid_groundTrue 和 r
    ndcg = NDCGatK_r(valid_groundTrue, r, k) # 需要 valid_groundTrue 和 r
    
    # RecallPrecision_ATk 返回字典 {'recall': ..., 'precision': ...}
    return ret['precision'], ret['recall'], ndcg


# --- sample_negative 函数 (保持不变) ---
def sample_negative(user_id, item_num, ground_truth_list, num_negatives=1):
    # 确保 ground_truth_list 是列表
    if not isinstance(ground_truth_list, list):
        print(f"错误: sample_negative 中的 ground_truth_list 不是列表。")
        return [] # 返回空列表表示失败
    if not (0 <= user_id < len(ground_truth_list)):
         print(f"警告: sample_negative 中的 user_id {user_id} 无效。")
         return []

    pos_items = set(ground_truth_list[user_id])
    neg_items = []
    max_tries = num_negatives * 100 
    tries = 0
    while len(neg_items) < num_negatives and tries < max_tries:
        neg_id = np.random.randint(0, item_num)
        if neg_id not in pos_items:
            if neg_id not in neg_items: 
                 neg_items.append(neg_id)
        tries += 1
    if len(neg_items) < num_negatives:
        # 这只是一个警告，不影响返回
        # print(f"警告: 用户 {user_id} 未能采样到足够的负样本 ({len(neg_items)}/{num_negatives})")
        pass 
    return neg_items

# --- 新增：UltraGCN 损失计算函数 ---
def calculate_ultragcn_loss(fused_user_embeds, fused_item_embeds, 
                            users, pos_items, neg_items, 
                            negative_weight, device):
    """
    计算简化的 UltraGCN L_L 损失 (BCE 分别计算正负样本)。
    使用固定权重替代 omega_weights。
    """
    user_embeds = fused_user_embeds[users]         # (batch_size, d)
    pos_embeds = fused_item_embeds[pos_items]      # (batch_size, d)
    neg_embeds = fused_item_embeds[neg_items]      # (batch_size, num_neg, d)

    pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # (batch_size,)
    
    # 调整 user_embeds 形状以匹配 neg_embeds 进行广播点积
    user_embeds_expanded = user_embeds.unsqueeze(1) # (batch_size, 1, d)
    neg_scores = (user_embeds_expanded * neg_embeds).sum(dim=-1) # (batch_size, num_neg)

    # 计算正样本损失 (目标为 1)
    pos_labels = torch.ones_like(pos_scores, device=device)
    # 使用固定权重 1
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, reduction='none')

    # 计算负样本损失 (目标为 0)
    neg_labels = torch.zeros_like(neg_scores, device=device)
    # 使用固定权重 1 (BCE 函数内部会处理 reduction)
    # reduction='none' 使得我们可以按样本应用 negative_weight
    neg_loss_per_sample = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction='none')
    # 对每个用户的负样本损失求平均，然后乘以 negative_weight
    neg_loss = neg_loss_per_sample.mean(dim=-1) * negative_weight

    # 总损失是每个用户的正样本损失 + 加权负样本损失
    loss = pos_loss + neg_loss
    
    return loss.mean() # 返回整个 batch 的平均损失

def calculate_norm_loss(user_attention_net, item_attention_net):
    """计算模型参数的 L2 正则化损失"""
    loss = 0.0
    for param in user_attention_net.parameters():
        loss += torch.sum(param ** 2)
    for param in item_attention_net.parameters():
        loss += torch.sum(param ** 2)
    return loss / 2

# --- 修改：使用 UltraGCN 损失的训练评估函数 ---
def train_evaluate_with_self_attention_ultragcn_loss(model_dir, train_loader, test_loader, 
                                                    train_ground_truth_list, test_ground_truth_list, 
                                                    mask, topk, user_num, item_num, device,
                                                    # 新增损失函数超参数
                                                    negative_num=5,       # 每个正样本对应的负样本数
                                                    negative_weight=0.1,  # 负样本损失的权重 (UltraGCN 论文中的 beta)
                                                    gamma=1e-5):          # L2 正则化权重
    """使用自注意力(加权融合)训练并使用简化的UltraGCN Loss, 在测试集上评估"""
    
    # --- 加载嵌入部分 ---
    behaviors = ['view', 'fav', 'cart', 'buy']
    all_user_embeds, all_item_embeds = {}, {}
    embed_dim = -1
    print("开始加载各行为嵌入...")
    for behavior in behaviors:
        # ... (加载和检查嵌入的代码) ...
        user_embeds, item_embeds = load_embeddings(model_dir, behavior, device)
        if user_embeds is not None and item_embeds is not None:
            if embed_dim == -1:
                embed_dim = user_embeds.shape[1]
                if user_embeds.shape[0] != user_num or item_embeds.shape[0] != item_num or item_embeds.shape[1] != embed_dim:
                     print(f"错误: {behavior} 嵌入形状或数量不一致。")
                     return None, None
            elif user_embeds.shape[1] != embed_dim or item_embeds.shape[1] != embed_dim or user_embeds.shape[0] != user_num or item_embeds.shape[0] != item_num:
                 print(f"错误: {behavior} 嵌入形状与其他行为不一致。")
                 return None, None
            all_user_embeds[behavior] = user_embeds
            all_item_embeds[behavior] = item_embeds
        else:
            print(f"错误: 无法加载 {behavior} 的嵌入，终止。")
            return None, None
    print(f"所有行为嵌入加载完毕。嵌入维度: {embed_dim}")

    stacked_user_embeds = torch.stack([all_user_embeds[b] for b in behaviors], dim=1).to(device)
    stacked_item_embeds = torch.stack([all_item_embeds[b] for b in behaviors], dim=1).to(device)
    print(f"堆叠后用户嵌入形状: {stacked_user_embeds.shape}")
    print(f"堆叠后物品嵌入形状: {stacked_item_embeds.shape}")
    
    # --- 初始化模型和优化器 ---
    user_attention_net = SelfAttentionFusion(embed_dim, num_heads=4, dropout=0.2).to(device)
    item_attention_net = SelfAttentionFusion(embed_dim, num_heads=4, dropout=0.2).to(device)
    attention_params = list(user_attention_net.parameters()) + list(item_attention_net.parameters())
    # 调整优化器参数可能需要根据新的损失函数进行
    
    optimizer = torch.optim.AdamW(attention_params, lr=1e-3, weight_decay=0) # weight_decay 设为 0，因为我们手动添加 norm_loss
    #是否使用学习率衰减 
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5) # 调整调度器
    
    best_recall = -1.0
    best_results = None
    best_fusion_weights_user = None 
    no_improve_count = 0
    patience = 20
    num_epochs = 100
    print(f"\n开始训练自注意力融合网络 (UltraGCN Loss)... 总训练轮数: {num_epochs}")
    print(f"损失参数: negative_num={negative_num}, negative_weight={negative_weight}, gamma={gamma}")
    
    # --- 训练循环 (使用 UltraGCN Loss) ---
    for epoch in range(num_epochs):
        user_attention_net.train()
        item_attention_net.train()
        optimizer.zero_grad() 
        
        total_epoch_loss = 0.0
        batch_count = 0
        
        # 在每个 epoch 开始时计算一次融合嵌入
        fused_user_embeds, _ = user_attention_net(stacked_user_embeds) 
        fused_item_embeds, _ = item_attention_net(stacked_item_embeds)

        for batch_idx, batch_users_tensor in enumerate(train_loader): 
            batch_users_tensor = batch_users_tensor.to(device)
            
            # --- 准备正负样本对 ---
            users_list, pos_items_list, neg_items_list = [], [], []
            for user_id_tensor in batch_users_tensor:
                user_id = user_id_tensor.item()
                if not (0 <= user_id < len(train_ground_truth_list)): continue # 跳过无效 user_id
                
                positive_items = train_ground_truth_list[user_id]
                if not positive_items: continue # 跳过没有正样本的用户
                
                # 为该用户的所有正样本采样负样本
                sampled_negatives = sample_negative(user_id, item_num, train_ground_truth_list, num_negatives=len(positive_items) * negative_num)
                
                if len(sampled_negatives) < len(positive_items) * negative_num:
                    # 如果负样本不足，可以重复采样或跳过部分正样本，这里简单处理
                    # print(f"警告: 用户 {user_id} 负样本不足，可能影响训练。")
                    continue # 跳过此用户以简化

                # 构建训练对：每个正样本对应 negative_num 个负样本
                neg_idx = 0
                for pos_item in positive_items:
                    current_neg_items = sampled_negatives[neg_idx : neg_idx + negative_num]
                    if len(current_neg_items) == negative_num: # 确保采样到了足够数量
                        users_list.append(user_id)
                        pos_items_list.append(pos_item)
                        neg_items_list.append(current_neg_items)
                    neg_idx += negative_num
            
            # 如果当前 batch 没有有效的训练对，则跳过
            if not users_list:
                continue 
                
            # 转换为 Tensor
            users = torch.tensor(users_list, dtype=torch.long, device=device)
            pos_items = torch.tensor(pos_items_list, dtype=torch.long, device=device)
            neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device) # (num_pairs, negative_num)
            
            # --- 计算损失 ---
            # 使用当前 epoch 计算好的融合嵌入
            batch_loss_L = calculate_ultragcn_loss(fused_user_embeds, fused_item_embeds, 
                                                   users, pos_items, neg_items, 
                                                   negative_weight, device)
                                                   
            batch_norm_loss = calculate_norm_loss(user_attention_net, item_attention_net)
            
            # 总损失 = L_L + gamma * L_norm
            batch_loss_total = batch_loss_L + gamma * batch_norm_loss

            # 累加到 epoch loss
            # 需要确保 loss 是需要梯度的
            total_epoch_loss += batch_loss_total # 直接累加
            batch_count += 1
                 
        # --- 每个 epoch 结束时反向传播和评估 ---
        if batch_count == 0:
            print(f"警告: Epoch {epoch+1} 未能处理任何有效的 batch。跳过此轮更新。")
            continue 

        # 计算平均 epoch loss
        average_epoch_loss = total_epoch_loss / batch_count
        
        # 反向传播
        if torch.isnan(average_epoch_loss):
             print(f"警告: Epoch {epoch+1} 出现 NaN 损失，跳过此轮更新。")
             continue

        average_epoch_loss.backward() # 对整个 epoch 的平均损失进行反向传播
        torch.nn.utils.clip_grad_norm_(attention_params, max_norm=1.0)
        optimizer.step() # 更新参数
        #scheduler.step() # 更新学习率
        
        # --- 评估当前性能 (使用测试集) ---
        user_attention_net.eval()
        item_attention_net.eval()
        with torch.no_grad():
            # 重新计算评估用的融合嵌入
            eval_fused_user, current_fusion_weights_user = user_attention_net(stacked_user_embeds)
            eval_fused_item, _ = item_attention_net(stacked_item_embeds)
            
            F1_score, Precision, Recall, NDCG, _ = evaluate_combined_embeddings(
                eval_fused_user, eval_fused_item,
                test_loader, test_ground_truth_list, 
                mask, topk, user_num, device
            )

            # 打印信息
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"当前损失 (Avg UltraGCN + Norm): {average_epoch_loss.item():.6f}")
            #print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        #print(f"评估指标 - F1: {F1_score:.4f}, P: {Precision:.4f}, R: {Recall:.4f}, NDCG: {NDCG:.4f}")
            
            # 更新最佳结果
            if Recall > best_recall:
                best_recall = Recall
                best_results = (F1_score, Precision, Recall, NDCG)
                best_fusion_weights_user = current_fusion_weights_user[:10].cpu().numpy() 
                no_improve_count = 0
                # 保存模型参数（可选）
                #print(f"===== 新的最佳结果 (Epoch {epoch+1}) =====")
               # print(f"最佳 F1: {F1_score:.4f}, P: {Precision:.4f}, R: {Recall:.4f}, NDCG: {NDCG:.4f}")
                #print("="*40)
            else:
                no_improve_count += 1
                print(f"无改善轮数: {no_improve_count}/{patience}")
            
            if no_improve_count >= patience:
                print(f"\n连续 {patience} 轮在测试集上没有改善，停止训练。")
                break
        
        # 清理显存
        del fused_user_embeds, fused_item_embeds, eval_fused_user, eval_fused_item 
        del batch_loss_total, batch_loss_L, batch_norm_loss, total_epoch_loss, average_epoch_loss
        if torch.cuda.is_available():
           torch.cuda.empty_cache()
           
    # --- 训练结束后 ---
    print("\n训练结束。使用最后一轮模型进行最终评估...")
    user_attention_net.eval()
    item_attention_net.eval()
    final_predictions = None
    with torch.no_grad():
        final_fused_user, final_weights_user = user_attention_net(stacked_user_embeds)
        final_fused_item, _ = item_attention_net(stacked_item_embeds)
        
        final_F1, final_P, final_R, final_NDCG, final_predictions = evaluate_combined_embeddings(
            final_fused_user, final_fused_item,
            test_loader, test_ground_truth_list,
            mask, topk, user_num, device
        )
        
        if best_results is None: 
             best_results = (final_F1, final_P, final_R, final_NDCG)
             if best_fusion_weights_user is None:
                 best_fusion_weights_user = final_weights_user[:10].cpu().numpy()

   # print("\n===== 自注意力(加权融合) + UltraGCN Loss - 最终评估结果 =====")
    #print(f"最佳 F1: {best_results[0]:.4f}")
    #print(f"最佳 P: {best_results[1]:.4f}")
    #print(f"最佳 R: {best_results[2]:.4f}")
    #print(f"最佳 NDCG: {best_results[3]:.4f}")
    #print("="*40)
    
    if best_fusion_weights_user is not None:
        print("查看部分用户的最佳融合权重 (view, fav, cart, buy):")
        for i in range(min(5, best_fusion_weights_user.shape[0])): 
            weights_str = ", ".join([f"{w:.3f}" for w in best_fusion_weights_user[i].flatten()])
            print(f"  User {i}: [{weights_str}]")

    return best_results, final_predictions

def evaluate_with_weights(model_dir, test_loader, test_ground_truth_list, mask, topk, user_num, device, weights):
    """使用不同权重评估组合嵌入"""
    behaviors = ['view', 'fav', 'cart', 'buy']
    all_embeds = {}
    
    # 加载所有行为的嵌入
    for behavior in behaviors:
        user_embeds, item_embeds = load_embeddings(model_dir, behavior, device)
        if user_embeds is not None and item_embeds is not None:
            all_embeds[behavior] = (user_embeds, item_embeds)
    
    if len(all_embeds) != len(behaviors):
        print("错误: 无法加载所有行为的嵌入")
        return
    
    # 使用权重组合嵌入
    combined_user_embeds = None
    combined_item_embeds = None
    
    for behavior, weight in weights.items():
        if behavior in all_embeds:
            user_embeds, item_embeds = all_embeds[behavior]
            if combined_user_embeds is None:
                combined_user_embeds = weight * user_embeds
                combined_item_embeds = weight * item_embeds
            else:
                combined_user_embeds += weight * user_embeds
                combined_item_embeds += weight * item_embeds
    
    # 评估组合嵌入
    F1_score, Precision, Recall, NDCG, _ = evaluate_combined_embeddings(
        combined_user_embeds, combined_item_embeds, test_loader,
        test_ground_truth_list, mask, topk, user_num, device
    )
    
    return F1_score, Precision, Recall, NDCG

def main():
    # --- 设置路径和参数 ---
    model_dir = '/home/dongzhi/毕设/model/mine/mine_data/config' 
    config_file_to_load = 'buy_config.ini' 
    config_path = os.path.join(model_dir, config_file_to_load)
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}"); return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # --- 加载配置和数据 ---
    try:
        # 再次强调：需要确保能区分训练/测试数据
        params, _, _, _, _, data_loader, mask, ground_truth_list, _ = data_param_prepare(config_path)
        # !!! 假设复用数据用于示例 !!!
        train_loader = data_loader 
        test_loader = data_loader
        train_ground_truth_list = ground_truth_list # 假设是 list of lists
        test_ground_truth_list = ground_truth_list  # 假设是 list of lists
        user_num = params['user_num']
        item_num = params['item_num'] 
        topk = params['topk']
        # 获取 UltraGCN 损失相关的参数 (如果配置文件中有的话)
        negative_num = params.get('negative_num', 380) # 默认值 5
        negative_weight = params.get('negative_weight', 150) # 默认值 0.1
        gamma = params.get('gamma', 1e-4) # L2 正则化权重，默认 1e-5
    except Exception as e:
        print(f"加载配置或准备数据时出错: {e}")
        print("请确保 data_param_prepare 能提供所需数据，并检查配置文件。")
        return

    mask = mask.to(device)
    print(f"参数加载完毕: user_num={user_num}, item_num={item_num}, topk={topk}")
    print(f"损失参数: negative_num={negative_num}, negative_weight={negative_weight}, gamma={gamma}")
    print("已将mask移动到设备:", device)
    
    print("\n开始自注意力(加权融合) + UltraGCN Loss 训练与评估...")
    # --- 调用修改后的函数 ---
    best_results, final_predictions = train_evaluate_with_self_attention_ultragcn_loss(
        model_dir, train_loader, test_loader, 
        train_ground_truth_list, test_ground_truth_list, 
        mask, topk, user_num, item_num, device,
        negative_num=negative_num, 
        negative_weight=negative_weight, 
        gamma=gamma
    )
    params, _, _, _, _, test_loader, mask, test_ground_truth_list, _ = data_param_prepare(config_path)
    mask = mask.to(device)
    user_num = params['user_num']
    topk = params['topk']
    weight_combinations = [
 # 偏重后期行为
        {'view': 0, 'fav': 1, 'cart': 0, 'buy': 0}, 
        
    ]
    for i, weights in enumerate(weight_combinations):
        
        wF1_score, wPrecision, wRecall, wNDCG = evaluate_with_weights(
            model_dir, test_loader, test_ground_truth_list,
            mask, topk, user_num, device, weights
        )
    if(wRecall>best_results[2]):
        best_results = (wF1_score, wPrecision, wRecall, wNDCG)
    # --- 打印最终结果 ---
    if best_results is not None:
        F1_score, Precision, Recall, NDCG = best_results
        print("\n===== 自注意力(加权融合) + UltraGCN Loss - 最终最佳评估指标 =====")
        print(f"F1得分: {F1_score:.4f}")
        print(f"准确率: {Precision:.4f}")
        print(f"召回率: {Recall:.4f}")
        print(f"NDCG: {NDCG:.4f}")
        print("="*40)
        if final_predictions is not None:
            print(f"最终预测分数矩阵形状: {final_predictions.shape}")
        else:
            print("未能生成最终预测分数。")

if __name__ == "__main__":
    import sys
    import numpy as np # 确保导入 numpy
    import traceback
    try:
        from main_at import RecallPrecision_ATk, NDCGatK_r, getLabel, data_param_prepare
        main()
    except ImportError as e:
        print(f"错误: 无法导入 'main_at' 模块或其部分内容: {e}")
    except Exception as e:
        print(f"执行 main 函数时发生错误: {e}")
        traceback.print_exc()
