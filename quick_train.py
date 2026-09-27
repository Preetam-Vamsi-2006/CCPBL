"""
Quick training script for sentiment analysis model
Trains on a sample of the IMDB dataset for faster iteration during development
"""

import subprocess
import sys

if __name__ == '__main__':
    print("Starting quick model training (25% sample)...\n")
    result = subprocess.run([sys.executable, 'train_model.py', '--sample'], cwd='.')
    sys.exit(result.returncode)
