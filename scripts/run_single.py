import os
import sys
from pathlib import Path

import torch

# StyleGAN2-ADA подключён как git submodule в корне проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLEGAN_PATH = PROJECT_ROOT / "stylegan2-ada-pytorch"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(STYLEGAN_PATH))

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

from configs.config import TrainConfig
from src.training.trainer import train_generator
from src.analysis.plots import full_analysis


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TrainConfig(
        experiment_name="baseline",
        save_dir="results",
        epochs=301,
    )

    df_metrics, df_grads, grad_norms_lst = train_generator(cfg=cfg, device=device)

    full_analysis(
        df_metrics=df_metrics,
        df_grads=df_grads,
        grad_norms_lst=grad_norms_lst,
        save_dir=cfg.save_dir,
        flag_heatmap=True,
    )


if __name__ == "__main__":
    main()
