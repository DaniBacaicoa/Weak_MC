import argparse
import json
import math
import os
import random
from pathlib import Path
from datetime import datetime
import inspect

import numpy as np
import torch
import torch.optim as optim
import pandas as pd

from utils.train_test_loop import train_and_evaluate
from utils.losses1 import MarginalChainProperLoss, ForwardProperLoss, scoring_matrix
from src.weakener import Weakener
from src.model import MLP
from src.dataset import Data_handling


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_output_dir(base: Path, args: argparse.Namespace) -> Path:
    run_dir = (
        base
        / "results"
        / args.dataset
        / f"{args.loss}-{args.loss_code}"
        / args.model_class
        / f"corr{args.corr_p:.2f}_seed{args.seed}_lr{args.lr}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def pick_loss(loss_name: str, M: np.ndarray, loss_code: str):
    loss_name = loss_name.strip()
    if loss_name == "MarginalChainProperLoss":
        return MarginalChainProperLoss(M, loss_code=loss_code)
    elif loss_name == "ForwardProperLoss":
        return ForwardProperLoss(M, loss_code=loss_code)
    else:
        raise ValueError(f"Unknown --loss '{loss_name}'. Choose MarginalChainProperLoss or ForwardProperLoss.")


def main():
    parser = argparse.ArgumentParser(description="Run weak-label training experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--model_class", type=str, default="pll", help="Weakener model for M generation (e.g., pll, unif_noise)")
    parser.add_argument("--corr_p", type=float, default=0.2, help="Correlation / noise parameter for Weakener.generate_M")
    parser.add_argument("--loss", type=str, default="MarginalChainProperLoss", choices=["MarginalChainProperLoss", "ForwardProperLoss"])
    parser.add_argument("--loss_code", type=str, default="cross_entropy", help="Loss code passed into the ProperLoss (e.g., cross_entropy, brier)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for Data_handling")
    parser.add_argument("--results_dir", type=str, default="results", help="Base directory to store outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lets do the steps in order:

    # 1) Data
    Data = Data_handling(
        dataset=args.dataset,
        train_size=0.8,
        test_size=0.2,
        batch_size=args.batch_size,
        shuffling=False,
        splitting_seed=args.seed,
    )

    # 2) Weakener + weak labels
    weakener = Weakener(true_classes=Data.num_classes)
    weakener.generate_M(model_class=args.model_class, corr_p=args.corr_p)
    M = weakener.M

    # Generate weak labels from ground truth one-hot
    true_onehot = Data.train_dataset.targets  # shape: (n_samples, n_classes)
    z, _ = weakener.generate_weak(true_onehot)
    Data.include_weak(z)  # Use weak labels in train set

    train_loader, test_loader = Data.get_dataloader(weak_labels="weak")

    # 3) Model & optimizer change it from here if you want to try different models
    model = MLP(
        input_size=Data.num_features,
        hidden_sizes=[512, 256],
        output_size=Data.num_classes,
        dropout_p=0.0,
        bn=False,
        activation="relu",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) Loss
    em_loss = pick_loss(args.loss, M, loss_code=args.loss_code)

    # 5) Train/eval

    model, results_df = train_and_evaluate(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        optimizer=optimizer,
        loss_fn=em_loss,
        num_epochs=args.epochs,
        corr_p=args.corr_p,
    )

    # 6) Save outputs
    run_dir = build_output_dir(Path(args.results_dir), args)

    # Save params
    params = {
        "seed": args.seed,
        "dataset": args.dataset,
        "lr": args.lr,
        "model_class": args.model_class,
        "corr_p": args.corr_p,
        "loss": args.loss,
        "loss_code": args.loss_code,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save results dataframe
    if isinstance(results_df, pd.DataFrame):
        results_df.to_csv(run_dir / "results.csv", index=False)
    else:
        # Try to coerce
        try:
            pd.DataFrame(results_df).to_csv(run_dir / "results.csv", index=False)
        except Exception:
            with open(run_dir / "results.csv", "w") as f:
                f.write("Could not serialize results_df\n")

    # Save model
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Save M
    np.save(run_dir / "M.npy", M)

    # Quick human-readable summary
    try:
        best_row = pd.read_csv(run_dir / "results.csv").sort_values("test_acc", ascending=False).iloc[0]
        summary = (
            f"Best epoch: {int(best_row['epoch'])}\n"
            f"train_loss={best_row.get('train_loss', float('nan'))}, "
            f"test_loss={best_row.get('test_loss', float('nan'))}, "
            f"train_acc={best_row.get('train_acc', float('nan'))}, "
            f"test_acc={best_row.get('test_acc', float('nan'))}\n"
        )
    except Exception:
        summary = "Summary unavailable (results.csv missing expected columns).\n"

    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)

    print(f"Saved run to: {run_dir}")

if __name__ == "__main__":
    main()
