import argparse
import subprocess
import sys


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Rebuild the main results table.")
    ap.add_argument("--use-cached", action="store_true",
                    help="skip training; rebuild tables from the committed results/raw")
    ap.add_argument("--fast", action="store_true",
                    help="2-trial smoke check of all models on one small dataset (minutes)")
    a = ap.parse_args()
    py = sys.executable

    if a.fast:
        run([py, "-m", "src.run", "--smoke", "--force", "--datasets", "brazil_houses",
             "--fractions", "0.1", "--seeds", "0"])
        return
    if not a.use_cached:
        run([py, "-m", "src.data"])
        run([py, "-m", "src.splits"])
        run([py, "-m", "src.run", "--fractions", "1.0"])  # all models, 100% data, 3 seeds
    run([py, "-m", "src.analyze"])


if __name__ == "__main__":
    main()