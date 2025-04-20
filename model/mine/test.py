import optuna
import torch
import configparser
import os
from main import (
    data_param_prepare, 
    UltraGCN, 
    test, 
    Sampling  # 添加Sampling函数的导入
)
import numpy as np

def objective(trial):
    config = configparser.ConfigParser()
    config.read('mine_data/config/buy_config.ini')
    
    # 模型参数
    config['Model']['initial_weight'] = str(trial.suggest_float('initial_weight', 1e-4, 1e-4, log=True))
    
    # 学习率
    config['Training']['learning_rate'] = str(trial.suggest_float('learning_rate', 5e-9, 5e-3, log=True))
    
    # w1-w4参数 - 参考当前配置调整搜索范围
    config['Training']['w1'] = str(trial.suggest_float('w1', 1e-7, 1, log=True))
    config['Training']['w2'] = str(trial.suggest_float('w2', 1e-7, 1))  # 因为当前是1
    config['Training']['w3'] = str(trial.suggest_float('w3', 1e-7, 1, log=True))
    config['Training']['w4'] = str(trial.suggest_float('w4', 1e-7, 1))  # 因为当前是1
    
    # negative sampling参数 - 根据当前配置调整
    config['Training']['negative_num'] = str(trial.suggest_int('negative_num', 900, 900))
    config['Training']['negative_weight'] = str(trial.suggest_int('negative_weight', 500, 500))
    
    # 正则化参数
    config['Training']['gamma'] = str(trial.suggest_float('gamma', 5e-7, 1, log=True))
    config['Training']['lambda'] = str(trial.suggest_float('lambda', 5e-7, 1))
    
    
    # 保存临时配置文件
    temp_config_path = 'temp_config4.ini'
    with open(temp_config_path, 'w') as f:
        config.write(f)
    
    try:
        # 准备数据和参数
        params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(temp_config_path)
        
        # 初始化模型
        model = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
        model = model.to(params['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        
        # 训练一定轮数进行评估
        best_recall = 0
        early_stop_count = 0
        
        for epoch in range(20):  # 为了快速验证，我们只训练20轮
            model.train()
            for batch, x in enumerate(train_loader):
                users, pos_items, neg_items = Sampling(x, params['item_num'], 
                                                     params['negative_num'], 
                                                     interacted_items, 
                                                     params['sampling_sift_pos'])
                users = users.to(params['device'])
                pos_items = pos_items.to(params['device'])
                neg_items = neg_items.to(params['device'])
                
                model.zero_grad()
                loss = model(users, pos_items, neg_items)
                loss.backward()
                optimizer.step()
            
            # 每5轮评估一次
            if epoch % 5 == 0:
                F1_score, Precision, Recall, NDCG = test(model, test_loader, 
                                                       test_ground_truth_list, 
                                                       mask, params['topk'], 
                                                       params['user_num'])
                if Recall > best_recall:
                    best_recall = Recall
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    
                if early_stop_count >= 2:  # 如果连续2次没有提升，则提前停止
                    break
                    
        # 返回最佳的Recall值作为优化目标
        return best_recall
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')
    
    finally:
        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # 进行50次试验
    
    print("最佳超参数:")
    print(study.best_params)
    print(f"最佳Recall@20: {study.best_value}")
    
    # 保存最佳参数到新的配置文件
    config = configparser.ConfigParser()
    config.read('mine_data/config/config_behavior_4.ini')
    
    # 更新配置文件中的最佳参数
    for param_name, param_value in study.best_params.items():
        if param_name in ['initial_weight']:
            config['Model'][param_name] = str(param_value)
        else:
            config['Training'][param_name] = str(param_value)
    
    # 保存优化后的配置文件
    with open('mine_data/config/config_behavior_4_optimized.ini', 'w') as f:
        config.write(f)

if __name__ == "__main__":
    main()