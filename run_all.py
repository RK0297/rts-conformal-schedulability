"""Automated end-to-end experiment pipeline for n=15 and n=20."""
import subprocess
import sys

def run_cmd(cmd):
    print(f"[*] Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        sys.exit(1)

def main():
    print("Starting automated pipeline...")

if __name__ == "__main__":
    main()
