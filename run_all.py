"""One-click automated runner for n=10 and n=15 (70% Presentation Milestone)."""
import subprocess
import sys

def run(cmd: str):
    print("\n" + "="*70)
    print(f"RUNNING: {cmd}")
    print("="*70 + "\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(1)

def main():
    # -------------------------------------------------
    # 1. Generate datasets (n=10 and n=15, 1000 per util)
    # -------------------------------------------------
    run("uv run python data/generate_tasksets.py --n 10 --num_per_util 1000")
    run("uv run python data/generate_tasksets.py --n 15 --num_per_util 1000")

    # -------------------------------------------------
    # 2. Train models (50 epochs, w=25, lr=0.001)
    # -------------------------------------------------
    run("uv run python training/train_baseline.py "
        "--data data/processed/tasksets_n10_uniform.npz "
        "--epochs 50 --batch_size 256 --lr 0.001 --w 25.0 "
        "--save_path experiments/results/joint_n10.pt")

    run("uv run python training/train_baseline.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--epochs 50 --batch_size 256 --lr 0.001 --w 25.0 "
        "--save_path experiments/results/joint_n15.pt")

    # -------------------------------------------------
    # 3. Run classical verifier
    # -------------------------------------------------
    run("uv run python verification/rta_verifier.py "
        "--data data/processed/tasksets_n10_uniform.npz "
        "--checkpoint experiments/results/joint_n10.pt")

    run("uv run python verification/rta_verifier.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--checkpoint experiments/results/joint_n15.pt")

    # -------------------------------------------------
    # 4. Calibrate Conformal Prediction
    # -------------------------------------------------
    run("uv run python training/calibrate_cp.py "
        "--data data/processed/tasksets_n10_uniform.npz "
        "--checkpoint experiments/results/joint_n10.pt "
        "--save_json experiments/results/calib_n10.json")

    run("uv run python training/calibrate_cp.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--checkpoint experiments/results/joint_n15.pt "
        "--save_json experiments/results/calib_n15.json")

    # -------------------------------------------------
    # 5. Sensitivity Analysis (Fig 9 Replication)
    # -------------------------------------------------
    run("uv run python evaluation/evaluate_weight_sensitivity.py")

    # -------------------------------------------------
    # 6. Evaluation sweeps & 2 master presentation figures
    # -------------------------------------------------
    run("uv run python evaluation/compare_baseline_vs_cp.py "
        "--data data/processed/tasksets_n10_uniform.npz "
        "--checkpoint experiments/results/joint_n10.pt")

    run("uv run python evaluation/compare_baseline_vs_cp.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--checkpoint experiments/results/joint_n15.pt")

    run("uv run python evaluation/generate_presentation_figures.py")

    print("\n" + "="*70)
    print("PIPELINE RUN COMPLETE FOR N=10 AND N=15 (70% MILESTONE READY)")
    print("="*70)

if __name__ == "__main__":
    main()