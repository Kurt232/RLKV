import io
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# --- Pre-define which methods to draw as lines ---

methods_to_show = ["h2o", "rkv", "duo_attn"]
# Define model pairs (reasoning, instruct)
model_pairs = [
    ("Llama-3.1-8B-R1", "Llama-3.1-8B-Inst"),
    ("Qwen-2.5-7B-R1", "Qwen-2.5-7B-Inst"),
    ("Qwen-3-4B-Thinking", "Qwen-3-4B-Instruct"),
]
datasets_to_plot = ["gsm8k", "math_500", "aime24", "mbpp"]

all_methods_in_plot = methods_to_show + ["full"]

# --- Define plot styling (matching result.png style) ---
style_map = {
    "h2o": {"color": "#1f77b4", "linestyle": "-", "marker": ".", "label": "H2O", "linewidth": 2.5, "markersize": 12},
    "rkv": {"color": "#ff7f0e", "linestyle": "-", "marker": "o", "label": "R-KV", "linewidth": 2.5, "markersize": 8},
    "duo_attn": {
        "color": "#d62728",
        "linestyle": "-",
        "marker": "x",
        "label": "DuoAttn",
        "linewidth": 2.5,
        "markersize": 8
    },
}

# Style for instruct models (dashed, more transparent)
instruct_style_map = {
    "h2o": {"color": "#1f77b4", "linestyle": "--", "marker": ".", "label": "H2O", "linewidth": 2.5, "markersize": 12, "alpha": 0.7},
    "rkv": {"color": "#ff7f0e", "linestyle": "--", "marker": "o", "label": "R-KV", "linewidth": 2.5, "markersize": 8, "alpha": 0.7},
    "duo_attn": {
        "color": "#d62728",
        "linestyle": "--",
        "marker": "x",
        "label": "DuoAttn",
        "linewidth": 2.5,
        "markersize": 8,
        "alpha": 0.7
    },
}


def plot_result(df, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Filter data to create a clean slate for plotting ---
    df_plot = df[df["method"].isin(all_methods_in_plot)]
    
    # Get all models from pairs
    all_models_to_show = []
    for reasoning, instruct in model_pairs:
        all_models_to_show.extend([reasoning, instruct])
    
    df_plot = df_plot[df_plot["model"].isin(all_models_to_show)]

    # --- Dynamically determine the plot grid and axes from the data ---
    dataset2name = {
        "gsm8k": "GSM8K (Math)",
        "math_500": "Math500 (Math)",
        "aime24": "AIME24 (Math)",
        "mbpp": "MBPP (Code)",
    }
    sparsity_ticks = np.unique(df_plot[df_plot["sparsity"] > 0]["sparsity"]).tolist()
    x_ticks = [0] + sparsity_ticks

    # 3. Create the figure and subplots adaptively
    if not datasets_to_plot or not model_pairs:
        print("No data available to plot after filtering.")
    else:
        total_subplots = len(model_pairs) * len(datasets_to_plot)
        ncols = 4 if total_subplots >= 4 else total_subplots
        nrows = (total_subplots + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3.8 * ncols, 3.5 * nrows),  # 从5.5x4.5减小到3.8x3.5
            sharex=True,
            sharey=False,
            squeeze=False,
        )

        # 4. Iterate through model pairs and datasets to populate each subplot
        for row_idx in range(nrows):
            for col_idx in range(ncols):
                subplot_idx = row_idx * ncols + col_idx
                if subplot_idx >= total_subplots:
                    axes[row_idx, col_idx].axis('off')  # Hide unused subplots
                    continue
                
                pair_idx = subplot_idx // len(datasets_to_plot)
                dataset_idx = subplot_idx % len(datasets_to_plot)
                
                reasoning_model, instruct_model = model_pairs[pair_idx]
                dataset_name = datasets_to_plot[dataset_idx]
                ax = axes[row_idx, col_idx]

                # Plot both reasoning and instruct models on the same subplot
                all_y_vals = []
                for model_type, model_name in [("reasoning", reasoning_model), ("instruct", instruct_model)]:
                    sub_df = df_plot[
                        (df_plot["model"] == model_name)
                        & (df_plot["dataset"] == dataset_name)
                    ]

                    # Get baseline (full method) score
                    baseline_row = sub_df[sub_df["method"] == "full"]
                    baseline_score = None
                    if not baseline_row.empty:
                        baseline_score = baseline_row["score"].iloc[0] / 100  # 转换为小数

                    # Choose style based on model type
                    current_style_map = style_map if model_type == "reasoning" else instruct_style_map

                    for method_name in methods_to_show:
                        method_df = sub_df[sub_df["method"] == method_name].sort_values("sparsity")

                        if not method_df.empty and baseline_score is not None:
                            # Calculate relative reduction (performance drop from full, negative values)
                            x_vals = method_df["sparsity"].tolist()
                            y_vals_raw = [score / 100 for score in method_df["score"].tolist()]
                            
                            # Calculate reduction as negative values, cap at 0 if performance > full
                            y_vals_reduction = [min(0, score - baseline_score) for score in y_vals_raw]
                            y_vals_reduction = [max(-1.0, val) for val in y_vals_reduction]

                            all_y_vals.extend(y_vals_reduction)
                            # Add zero point at sparsity 0
                            x_vals = [0] + x_vals
                            y_vals_reduction = [0] + y_vals_reduction

                            style_kwargs = current_style_map[method_name].copy()

                            # Add label only for reasoning model to avoid duplicate legends
                            if model_type == "reasoning":
                                style_kwargs["label"] = style_map[method_name]["label"]
                            
                            ax.plot(
                                x_vals,
                                y_vals_reduction,
                                **style_kwargs
                            )

                # 5. Format each subplot
                model_pair_name = f"{'-'.join(reasoning_model.split('-')[:-1])} series"
                ax.set_title(f"{model_pair_name}\n{dataset2name.get(dataset_name, dataset_name)}", fontsize=14, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel("Performance Drop", fontsize=14, fontweight='bold')
                if col_idx == len(datasets_to_plot) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(" ", fontsize=14, fontweight='bold') # 右侧留空以对齐

                # 美化网格
                ax.grid(True, which="both", linestyle="-", linewidth=0.6, alpha=0.7)
                ax.set_axisbelow(True)
                
                ax.set_xticks(x_ticks)
                ax.tick_params(axis='x', labelbottom=True, labelsize=12)  # 增大刻度标签字体
                ax.tick_params(axis='y', labelsize=12)  # 增大刻度标签字体
                
                # 更紧凑的x轴范围，因为只有5个点
                ax.set_xlim(-0.05, max(x_ticks) + 0.05)  # 减小边距
                
                # Set y-axis from 0 (top) to -1 (bottom) with 0.1 intervals

                if all_y_vals:
                    y_min = min(all_y_vals)
                    y_max = max(all_y_vals)
                    
                    y_max_tick = 0.0
                    
                    # 向下取到最近的0.1倍数
                    y_min_tick = -np.ceil(-y_min * 10) / 10
                    y_min_tick = max(y_min_tick, -1.0)  # 不低于-1.0
                    
                    # 如果最小值太接近于0.1的网格线，再往下延长0.1
                    # 检查实际最小值与最近网格线的距离
                    nearest_grid_line = -np.ceil(-y_min * 10) / 10
                    distance_to_grid = y_min - nearest_grid_line
                    
                    # 如果距离小于0.03（太接近网格线），向下延长0.1
                    if distance_to_grid < 0.03:
                        y_min_tick -= 0.1
                        y_min_tick = max(y_min_tick, -1.0)  # 确保不超出-1.0的限制

                    # 确保最小范围为0.2以保证可读性
                    if y_max_tick - y_min_tick < 0.2:
                        y_min_tick = y_max_tick - 0.2
                    
                    # 生成以0.1为间隔的y刻度
                    y_ticks = np.arange(y_min_tick, y_max_tick + 0.05, 0.1)  # +0.05是为了包含上边界
                    y_ticks = np.round(y_ticks, 1)  # 避免浮点精度问题
                    
                    ax.set_ylim(y_min_tick, y_max_tick)
                    ax.set_yticks(y_ticks)

        # 6. Finalize the figure with a shared legend and labels - 增大图例字体
        legend_handles = [
            plt.Line2D([0], [0], 
                      color=style_map[key]["color"],
                      linestyle=style_map[key]["linestyle"],
                      marker=style_map[key].get("marker", None),
                      markersize=style_map[key].get("markersize", 8),
                      linewidth=style_map[key]["linewidth"],
                      label=style_map[key]["label"]) 
            for key in methods_to_show
        ]
        
        # Add legend entries for reasoning vs instruct
        reasoning_handle = plt.Line2D([0], [0], color="black", linestyle="-", label="Reasoning", markersize=8)
        instruct_handle = plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.7, label="Instruct", markersize=8)
        
        all_handles = legend_handles + [reasoning_handle, instruct_handle]
        all_labels = [h.get_label() for h in all_handles]

        # Use result.png legend style - single row, top placement
        fig.legend(
            handles=all_handles,
            labels=all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(all_handles),
            frameon=False,
            fontsize=16,
        )

        fig.tight_layout(pad=2.5, rect=[0, 0, 1, 0.98], h_pad=1.0)  # 更紧凑的行间距和下方空间
        # 添加整个图的x轴标签，位置更靠下
        fig.text(0.5, 0.005, "KV Cache Budget Sparsity", ha='center', va='bottom', 
            fontsize=14, fontweight='bold')
        # 保存高质量图片
        fig.savefig(os.path.join(out_dir, "motivation_full.png"), 
            dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
        fig.savefig(os.path.join(out_dir, "motivation_full.pdf"), 
            bbox_inches='tight', 
            facecolor='white', edgecolor='none')  # 额外保存PDF版本供论文使用
        plt.close(fig)


# Load data
df_list = []
for m in models:
    all_models_to_show = []
    for reasoning, instruct in model_pairs:
        all_models_to_show.extend([reasoning, instruct])
    
    if all_models_to_show and m not in all_models_to_show:
        print(f"Skipping {m}...")
        continue
    json_data = json.load(open(osp.join(root, "pred", m, "result.json")))
    df = pd.DataFrame(
        [
            {
                "model": m,
                "dataset": k.split("-")[0].rstrip("_c"),
                "method": k.replace(".jsonl", "").split("-")[1],
                "sparsity": float(k[k.find("-sp-") + 4 : -6]) if "-sp-" in k else 0.0,
                "score": v,
                "expr": k.replace(".jsonl", ""),
            }
            for k, v in json_data.items()
        ]
    )
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
plot_result(df, out_dir=out_dir)