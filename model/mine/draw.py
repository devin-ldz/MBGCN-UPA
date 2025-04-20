import json
import matplotlib.pyplot as plt

# 读取 JSON 文件
file_path = "/home/dongzhi/毕设/model/mine/param_tuning_results_20250413_114854/experiment_results.json"
with open(file_path, "r") as f:
    data = json.load(f)

# 通用画图函数
def plot_results(results, param_name, title, save_path, log_x=False):
    params = results["params"]
    recall_main = [r["main"]["recall"] for r in results["results"]]
    recall_at = [r["main_at"]["recall"] for r in results["results"]]
    ndcg_main = [r["main"]["ndcg"] for r in results["results"]]
    ndcg_at = [r["main_at"]["ndcg"] for r in results["results"]]

    # 👉 根据参数排序所有指标
    combined = list(zip(params, recall_main, recall_at, ndcg_main, ndcg_at))
    combined.sort(key=lambda x: x[0])  # 按参数值升序

    # 解包排序后的数据
    params, recall_main, recall_at, ndcg_main, ndcg_at = zip(*combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall
    axes[0].plot(params, recall_main, label="ultragcn", marker="o")
    axes[0].plot(params, recall_at, label="ultragcn-cate", marker="s")
    axes[0].set_title(f"{title} - Recall")
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Recall")
    if log_x:
        axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(True)

    # NDCG
    axes[1].plot(params, ndcg_main, label="ultragcn", marker="o")
    axes[1].plot(params, ndcg_at, label="ultragcn-cate", marker="s")
    axes[1].set_title(f"{title} - NDCG")
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("NDCG")
    if log_x:
        axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"保存图像至: {save_path}")

# 输出图像路径
base_dir = "/home/dongzhi/毕设/model/mine/param_tuning_results_20250413_114854"
gamma_plot_path = f"{base_dir}/gamma_param_tuning.png"
lambda_plot_path = f"{base_dir}/lambda_param_tuning.png"

# 绘图并保存
plot_results(data["gamma_results"], "Gamma", "Gamma Parameter Tuning", gamma_plot_path, log_x=True)
plot_results(data["lambda_results"], "Lambda", "Lambda Parameter Tuning", lambda_plot_path, log_x=False)