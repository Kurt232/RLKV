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

note = None
if len(sys.argv) == 2:
    note = sys.argv[1]

models = [f for f in os.listdir(osp.join(root, "pred"))]

# 2. Pre-process and configure the plot
# --- Pre-define which methods to draw as bars ---
methods_to_show = ["h2o", "rkv", "duo_attn"]
all_methods_in_plot = methods_to_show
models_to_show = [ 
    "Llama-3.1-8B-R1",
    "Qwen-2.5-7B-R1",
    "Qwen-3-4B-Thinking",
    ]
# --- Define plot styling ---
style_map = {
    "h2o": {"color": "#1f77b4", "label": "H2O"},
    "rkv": {"color": "#ff7f0e", "label": "R-KV"},
    "duo_attn": {"color": "#d62728", "label": "DuoAttn"},
    # "rlkv": {"color": "#9370DB", "label": "Ours"}
}

# 定义错误类型的颜色深浅（从深到浅：repeat->incorrect->overlength）
error_type_alphas = {
    "repeat_rate": 1.0,      # 最深色（底部）
    "incorrect_rate": 0.4,   # 中等深度（中间）
    "overlength_rate": 0.7   # 最浅色（顶部）
}

def plot_result(df, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Filter data to create a clean slate for plotting ---
    df_plot = df[df["method"].isin(all_methods_in_plot)]
    if models_to_show is not None:
        df_plot = df_plot[df_plot["model"].isin(models_to_show)]

    # --- Dynamically determine the plot grid and axes from the data ---
    models_to_plot = models_to_show
    datasets_to_plot = ["gsm8k", "math_500", "aime24", "mbpp"]
    dataset2name = {
        "gsm8k": "GSM8K (Math)",
        "math_500": "Math500 (Math)",
        "aime24": "AIME24 (Math)",
        "mbpp": "MBPP (Code)",
    }
    sparsity_levels = [0.2, 0.4, 0.6, 0.8]

    # 3. Create the figure and subplots adaptively
    if not datasets_to_plot or not models_to_plot:
        print("No data available to plot after filtering.")
    else:
        fig, axes = plt.subplots(
            nrows=len(models_to_plot),
            ncols=len(datasets_to_plot),
            figsize=(3.5 * len(datasets_to_plot), 3.5 * len(models_to_plot)),
            sharex=True,
            sharey=False,
            squeeze=False,
        )

        # 4. Iterate through the discovered models (rows) and datasets (columns) to populate each subplot
        for row_idx, model_name in enumerate(models_to_plot):
            for col_idx, dataset_name in enumerate(datasets_to_plot):
                ax = axes[row_idx, col_idx]
                sub_df = df_plot[
                    (df_plot["model"] == model_name)
                    & (df_plot["dataset"] == dataset_name)
                ]

                # 设置柱状图参数
                bar_width = 0.2
                n_methods = len(all_methods_in_plot)
                x_positions = np.arange(len(sparsity_levels))
                
                # 为每个方法绘制堆叠柱状图
                for method_idx, method_name in enumerate(all_methods_in_plot):
                    method_df = sub_df[sub_df["method"] == method_name]
                    
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

                # 5. Format each subplot
                ax.set_title(f"{model_name}\n{dataset2name.get(dataset_name, dataset_name)}", 
                           fontsize=14, fontweight='bold')
                
                # 每个子图都设置y轴标签为Error Rate
                if col_idx == 0:
                    ax.set_ylabel("Error Rate", fontsize=14)
                elif col_idx == len(datasets_to_plot) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(" ", fontsize=14)
                
                # 美化网格
                ax.grid(True, which="both", linestyle="-", linewidth=0.6, alpha=0.7, axis='y')
                ax.set_axisbelow(True)
                
                # 设置x轴
                ax.set_xticks(x_positions)
                ax.set_xticklabels([f"{sp:.1f}" for sp in sparsity_levels])
                ax.tick_params(axis='x', labelbottom=True, labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                
                # 设置y轴范围 - 参照原代码的0-1范围，间隔0.1
                ax.set_ylim(0, 1.0)
                ax.set_yticks(np.arange(0, 1.1, 0.1))

        # 6. 创建双重图例 - 都放在图像最上方
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
        
        # 放置图例 - 都在顶部
        method_legend = fig.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(method_handles),
            frameon=False,
            fontsize=14,
        )
        
        # error_legend = fig.legend(
        #     handles=error_handles,
        #     loc="upper center", 
        #     bbox_to_anchor=(0.5, 0.94),
        #     ncol=len(error_handles),
        #     frameon=False,
        #     fontsize=14,
        # )
        
        # 添加第一个图例到图中，避免被第二个覆盖
        fig.add_artist(method_legend)

        # 调整布局 - 为顶部图例留出更多空间
        fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.91], h_pad=1.0)
        
        # 添加整个图的x轴标签
        fig.text(0.5, 0.005, "KV Cache Budget Sparsity", ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
        
        # 保存高质量图片
        fig.savefig(os.path.join(out_dir, f"error_full.png"), 
                   dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        fig.savefig(os.path.join(out_dir, f"error_full.pdf"), 
                   bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)


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
plot_result(df, out_dir=out_dir)