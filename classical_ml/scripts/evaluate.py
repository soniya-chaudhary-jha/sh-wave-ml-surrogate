import subprocess
import sys

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "ml.evaluate_model"]
    subprocess.run(cmd, check=True)
    print("Evaluation finished.")
