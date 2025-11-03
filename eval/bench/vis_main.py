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
# --- Pre-define which methods to draw as lines ---
methods_to_show = ["h2o", "rkv", "duo_attn", "rlkv"]
all_methods_in_plot = methods_to_show + ["full"]
models_to_show = [ 
    "Llama-3.1-8B-R1",
    "Qwen-2.5-7B-R1",
    "Qwen-3-4B-Thinking",
    ]
# --- Define plot styling ---
style_map = {
    "Full": {"color": "gray", "linestyle": "--", "label": "Full", "linewidth": 2.5},
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
    "rlkv": {
        "color": "#9370DB",
        "linestyle": "-",
        "marker": "s",
        "label": "Ours",
        "linewidth": 2.5,
        "markersize": 8
    }
}


def plot_result(df, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Filter data to create a clean slate for plotting ---
    df_plot = df[df["method"].isin(all_methods_in_plot)]
    if models_to_show is not None:
        df_plot = df_plot[df_plot["model"].isin(models_to_show)]

    # --- Dynamically determine the plot grid and axes from the data ---
    models_to_plot = models_to_show
    # datasets_to_plot = sorted(df_plot["dataset"].unique())
    datasets_to_plot = ["gsm8k", "math_500", "aime24", "mbpp"]
    dataset2name = {
        "gsm8k": "GSM8K (Math)",
        "math_500": "Math500 (Math)",
        "aime24": "AIME24 (Math)",
        "mbpp": "MBPP (Code)",
    }
    sparsity_ticks = np.unique(df_plot[df_plot["sparsity"] > 0]["sparsity"]).tolist()
    x_ticks = [0] + sparsity_ticks

    # 3. Create the figure and subplots adaptively - 更紧凑的尺寸
    if not datasets_to_plot or not models_to_plot:
        print("No data available to plot after filtering.")
    else:
        # 减小图表尺寸，因为x轴只有5个点
        fig, axes = plt.subplots(
            nrows=len(models_to_plot),
            ncols=len(datasets_to_plot),
            figsize=(3.8 * len(datasets_to_plot), 3.5 * len(models_to_plot)),  # 从5.5x4.5减小到3.8x3.5
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

                baseline_row = sub_df[sub_df["method"] == "full"]
                baseline_score = None
                if not baseline_row.empty:
                    baseline_score = baseline_row["score"].iloc[0] / 100  # 转换为小数
                    ax.axhline(
                        baseline_score,
                        color=style_map["Full"]["color"],
                        linestyle=style_map["Full"]["linestyle"],
                        linewidth=style_map["Full"]["linewidth"],
                    )

                # Collect all y values for this subplot to determine y-axis range
                all_y_vals = []
                if baseline_score is not None:
                    all_y_vals.append(baseline_score)

                for method_name in methods_to_show:
                    method_df = sub_df[sub_df["method"] == method_name].sort_values(
                        "sparsity"
                    )
                    if not method_df.empty:
                        x_vals = method_df["sparsity"].tolist()
                        y_vals = [score / 100 for score in method_df["score"].tolist()]  # 转换为小数
                        all_y_vals.extend(y_vals)
                        
                        if baseline_score is not None:
                            y_vals = [baseline_score] + y_vals
                            x_vals = [0] + x_vals

                        ax.plot(
                            x_vals,
                            y_vals,
                            color=style_map[method_name]["color"],
                            linestyle=style_map[method_name]["linestyle"],
                            marker=style_map[method_name]["marker"],
                            markersize=style_map[method_name]["markersize"],
                            linewidth=style_map[method_name]["linewidth"],
                        )

                # 5. Format each subplot: set column titles and row labels - 增大字体
                ax.set_title(f"{dataset2name.get(dataset_name, dataset_name)}", 
                           fontsize=14, fontweight='bold')  # 增大标题字体
                
                # 只在每行的第一列设置y轴标签为模型名称
                if col_idx == 0:
                    ax.set_ylabel(model_name, fontsize=14, fontweight='bold')
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
                
                # Set adaptive y-axis with ticks at intervals of 0.1, starting from 0.0
                if all_y_vals:
                    y_min = min(all_y_vals)
                    y_max = max(all_y_vals)
                    
                    # 最低点始终为0.0
                    y_min_tick = 0.0
                    
                    # 向上取到最近的0.1倍数
                    y_max_tick = np.ceil(y_max * 10) / 10
                    if y_max_tick - y_max < 0.05:
                        y_max_tick += 0.1
                    y_max_tick = min(y_max_tick, 1.0)  # 不超过1.0
                    
                    # 确保最小范围为0.2以保证可读性
                    if y_max_tick - y_min_tick < 0.2:
                        y_max_tick = y_min_tick + 0.2
                    
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
            for key in ["Full"] + methods_to_show
        ]
        
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),  # 稍微调整位置
            ncol=len(style_map),
            frameon=False,
            fontsize=15,  # 增大图例字体
            columnspacing=1.5,  # 增加列间距
        )

        # 调整布局，更紧凑的间距
    fig.tight_layout(pad=2.5, rect=[0, 0, 1, 0.95], h_pad=1.0)  # 更紧凑的行间距和下方空间
    # 添加整个图的x轴标签，位置更靠下
    fig.text(0.5, 0.005, "KV Cache Budget Sparsity", ha='center', va='bottom', 
        fontsize=14, fontweight='bold')
    # 保存高质量图片
    fig.savefig(os.path.join(out_dir, f"main_result.jpg"), 
        dpi=300, bbox_inches='tight', 
        facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(out_dir, f"main_result.pdf"), 
        bbox_inches='tight', 
        facecolor='white', edgecolor='none')  # 额外保存PDF版本供论文使用
    plt.close(fig)


df_list = []
for m in models:
    if models_to_show and m not in models_to_show:
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