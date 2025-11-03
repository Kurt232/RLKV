import io
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置更好的字体和样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.5

root = osp.join(osp.dirname(__file__))
out_dir = osp.join(root, "figs")
os.makedirs(out_dir, exist_ok=True)

models = [f for f in os.listdir(osp.join(root, "pred"))]

# 2. Pre-process and configure the plot
# --- Pre-define which methods to draw as bars ---
methods_to_show = ["rkv", "duo_attn"]
all_methods_in_plot = methods_to_show
target_dataset = "mbpp"  # 只选择Math500数据集
model_name = "Llama-3.1-8B-R1"
models_to_show = [model_name]  # 只选择一个模型
# --- Define plot styling --- (保持完全一致)
style_map = {
    "h2o": {"color": "#1f77b4", "label": "H2O"},
    "rkv": {"color": "#ff7f0e", "label": "Token-dropping"},
    "duo_attn": {"color": "#d62728", "label": "Head-reallocating"},
    "rlkv": {"color": "#9370DB", "label": "RLKV (Ours)"}
}

# 定义错误类型的颜色深浅（从深到浅：repeat->incorrect->overlength）
error_type_alphas = {
    "repeat_rate": 1.0,      # 最深色（底部）
    "incorrect_rate": 0.4,   # 中等深度（中间）
    "overlength_rate": 0.7   # 最浅色（顶部）
}

def plot_single_result(df, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Filter data to create a clean slate for plotting ---
    df_plot = df[df["method"].isin(all_methods_in_plot)]
    df_plot = df_plot[df_plot["model"] == model_name]
    df_plot = df_plot[df_plot["dataset"] == target_dataset]

    dataset2name = {
        "math_500": "Math500 (Math)",
        "mbpp": "MBPP (Code)",
    }
    sparsity_levels = [0.2, 0.4, 0.6, 0.8]

    # 3. Create single figure
    if df_plot.empty:
        print("No data available to plot after filtering.")
        return
    
    # 创建单个子图，尺寸适中
    fig, ax = plt.subplots(figsize=(3.8, 3.5))

    # 4. 绘制数据
    # 设置柱状图参数
    bar_width = 0.2
    n_methods = len(all_methods_in_plot)
    x_positions = np.arange(len(sparsity_levels))
    
    # 为每个方法绘制堆叠柱状图
    for method_idx, method_name in enumerate(all_methods_in_plot):
        method_df = df_plot[df_plot["method"] == method_name]
        
        # 计算每个柱子的x位置
        x_offset = (method_idx - n_methods / 2 + 0.5) * bar_width
        x_pos = x_positions + x_offset
        
        # 准备数据
        repeat_rates = []
        incorrect_rates = []
        overlength_rates = []
        
        for sparsity in sparsity_levels:
            row = method_df[method_df["sparsity"] == sparsity]
            
            if not row.empty:
                repeat_rates.append(row["repeat_rate"].iloc[0] / 100.0)  # 转换为小数
                incorrect_rates.append(row["incorrect_rate"].iloc[0] / 100.0)
                overlength_rates.append(row["overlength_rate"].iloc[0] / 100.0)
            else:
                repeat_rates.append(0)
                incorrect_rates.append(0)
                overlength_rates.append(0)
        
        # 绘制堆叠柱状图
        base_color = style_map[method_name]["color"]
        
        # 只在有数据的位置绘制柱子
        valid_indices = []
        valid_x_pos = []
        valid_repeat = []
        valid_incorrect = []
        valid_overlength = []
        
        for i, (rep, inc, over) in enumerate(zip(repeat_rates, incorrect_rates, overlength_rates)):
            if rep + inc + over > 0:  # 只有当总错误率大于0时才绘制
                valid_indices.append(i)
                valid_x_pos.append(x_pos[i])
                valid_repeat.append(rep)
                valid_incorrect.append(inc)
                valid_overlength.append(over)
        
        if valid_repeat:  # 确保有数据才绘制
            # 底部：repeat_rate
            ax.bar(valid_x_pos, valid_repeat, bar_width, 
                  color=base_color, alpha=error_type_alphas["repeat_rate"],
                  edgecolor='white', linewidth=0.5)
            
            # 中间：incorrect_rate  
            ax.bar(valid_x_pos, valid_incorrect, bar_width,
                  bottom=valid_repeat,
                  color=base_color, alpha=error_type_alphas["incorrect_rate"],
                  edgecolor='white', linewidth=0.5)
            
            # 顶部：overlength_rate
            bottom_vals = [r + i for r, i in zip(valid_repeat, valid_incorrect)]
            ax.bar(valid_x_pos, valid_overlength, bar_width,
                  bottom=bottom_vals,
                  color=base_color, alpha=error_type_alphas["overlength_rate"],
                  edgecolor='white', linewidth=0.5)

    # 5. Format the subplot (保持样式一致)
    ax.set_title(f"{model_name} - {dataset2name.get(target_dataset, target_dataset)}", 
               fontsize=9, fontweight='bold')
    
    ax.set_ylabel("Error Rate", fontsize=9, fontweight='bold')
    
    # 美化网格
    ax.grid(True, which="both", linestyle="-", linewidth=0.6, alpha=0.7, axis='y')
    ax.set_axisbelow(True)
    
    # 设置x轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{sp:.1f}" for sp in sparsity_levels])
    ax.tick_params(axis='x', labelbottom=True, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel("KV Cache Budget Sparsity", fontsize=9, fontweight='bold')
    
    # 设置y轴范围 - 参照原代码的0-1范围，间隔0.1
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # 6. 创建双重图例 - 都放在图像最上方 (保持样式一致)
    # 方法图例 - 第一排
    method_handles = [
        plt.Rectangle((0,0),1,1, facecolor=style_map[method]["color"], 
                     alpha=0.8, label=style_map[method]["label"])
        for method in all_methods_in_plot
    ]
    
    # 错误类型图例 - 第二排
    error_handles = [
        plt.Rectangle((0,0),1,1, facecolor='gray', alpha=error_type_alphas["repeat_rate"], 
                     label='Repeat Rate'),
        plt.Rectangle((0,0),1,1, facecolor='gray', alpha=error_type_alphas["incorrect_rate"], 
                     label='Incorrect Rate'),
        plt.Rectangle((0,0),1,1, facecolor='gray', alpha=error_type_alphas["overlength_rate"], 
                     label='Overlength Rate')
    ]
    
    # 放置图例 - 缩小字体和间距
    method_legend = fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(method_handles),
        frameon=False,
        fontsize=9,  # 从15减小到11
        columnspacing=0.8,  # 减小列间距
    )

    # error_legend = fig.legend(
    #     handles=error_handles,
    #     loc="upper center", 
    #     bbox_to_anchor=(0.5, 0.94),
    #     ncol=len(error_handles),
    #     frameon=False,
    #     fontsize=10,  # 从15减小到10
    #     columnspacing=0.8,  # 减小列间距
    # )
    
    # 添加第一个图例到图中，避免被第二个覆盖
    fig.add_artist(method_legend)

    # 调整布局 - 为顶部图例留出更多空间
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # 保存高质量图片
    fig.savefig(os.path.join(out_dir, f"error.png"), 
               dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(out_dir, f"error.pdf"), 
               bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)


# 数据读取部分保持一致
df_list = []
for m in models:
    if models_to_show and m not in models_to_show:
        print(f"Skipping {m}...")
        continue
    json_data = json.load(open(osp.join(root, "pred", m, "result_evals.json")))["results"]
    df = pd.DataFrame(
        [
            {
                "model": m,
                "dataset": k.split("-")[0].rstrip("_c"),
                "method": k.replace(".jsonl", "").split("-")[1],
                "sparsity": float(k[k.find("-sp-") + 4 : -6]) if "-sp-" in k else 0.0,
                "expr": k.replace(".jsonl", ""),
                "error_rate": v[0],
                "incorrect_rate": v[1],
                "overlength_rate": v[2],
                "repeat_rate": v[3],
            }
            for k, v in json_data.items()
        ]
    )
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
plot_single_result(df, out_dir=out_dir)