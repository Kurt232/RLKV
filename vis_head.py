import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def visualize_pruned_attention_heads(full_attention_heads, fig_name):
    img = np.array(full_attention_heads)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="coolwarm", interpolation="nearest")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.colorbar(fraction=0.046, pad=0.04)
    # scale the color to 0-1
    plt.clim(0, 1)
    plt.tight_layout()
    plt.title("Ratio of Full Attention Computations")
    
    fig.savefig(fig_name)

if __name__ == "__main__":
    root = Path("head_dist")
    for expr_dir in root.glob("*"):
        print(f"Processing {expr_dir}")
        for model_dir in expr_dir.glob("*"):
            for trail in model_dir.glob("*"):
                path = trail / "adapter_weights.tsv"    
                if not path.exists():
                    path = trail / "full_attention_heads.tsv"
                    if not path.exists():
                        print(f"Skip {path}: not found")
                        continue
                data = np.loadtxt(path, delimiter="\t")

                fig_name = trail / f"{trail.name}.png"
                if not fig_name.exists():
                    visualize_pruned_attention_heads(data, fig_name)
                    print(f"Saving figure to {fig_name}")
                stats_path = trail / "stats.txt"
                # if not stats_path.exists():
                #     mean = data.mean()
                #     std = data.std()
                #     open(trail / "stats.txt", "w").write(f"Mean: {mean}\nStd: {std}\n")