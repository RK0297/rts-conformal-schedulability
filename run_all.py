"""One-click automated runner for n=15 and n=20 with 100 epochs, w=25, and 128->64->64 MLP."""
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
    # 1. Generate datasets (n=15 and n=20, 1500 per util)
    # -------------------------------------------------
    run("uv run python data/generate_tasksets.py --n 15 --num_per_util 1500")
    run("uv run python data/generate_tasksets.py --n 20 --num_per_util 1500")

    # -------------------------------------------------
    # 2. Train models (100 epochs, w=25, lr=0.001)
    # -------------------------------------------------
    run("uv run python training/train_baseline.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--epochs 100 --batch_size 256 --lr 0.001 --w 25.0 "
        "--save_path experiments/results/joint_n15.pt")

    run("uv run python training/train_baseline.py "
        "--data data/processed/tasksets_n20_uniform.npz "
        "--epochs 100 --batch_size 256 --lr 0.001 --w 25.0 "
        "--save_path experiments/results/joint_n20.pt")

    # -------------------------------------------------
    # 3. Run classical verifier
    # -------------------------------------------------
    run("uv run python verification/rta_verifier.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--checkpoint experiments/results/joint_n15.pt")

    run("uv run python verification/rta_verifier.py "
        "--data data/processed/tasksets_n20_uniform.npz "
        "--checkpoint experiments/results/joint_n20.pt")

    # -------------------------------------------------
    # 4. Calibrate Conformal Prediction
    # -------------------------------------------------
    run("uv run python training/calibrate_cp.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--checkpoint experiments/results/joint_n15.pt "
        "--save_json experiments/results/calib_n15.json")

    run("uv run python training/calibrate_cp.py "
        "--data data/processed/tasksets_n20_uniform.npz "
        "--checkpoint experiments/results/joint_n20.pt "
        "--save_json experiments/results/calib_n20.json")

    # -------------------------------------------------
    # 5. Final evaluation + plots
    # -------------------------------------------------
    run("uv run python evaluation/compare_baseline_vs_cp.py "
        "--data data/processed/tasksets_n15_uniform.npz "
        "--checkpoint experiments/results/joint_n15.pt")

    run("uv run python evaluation/compare_baseline_vs_cp.py "
        "--data data/processed/tasksets_n20_uniform.npz "
        "--checkpoint experiments/results/joint_n20.pt")

    print("\n" + "="*70)
    print("ALL RUNS COMPLETE FOR N=15 AND N=20")
    print("="*70)

if __name__ == "__main__":
    main()