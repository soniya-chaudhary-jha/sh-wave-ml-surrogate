import subprocess
import sys

if __name__ == "__main__":
    # Run the training module which trains and saves model+scaler
    cmd = [sys.executable, "-m", "ml.train_surrogate"]
    subprocess.run(cmd, check=True)
    print("Training finished.")
