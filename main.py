import os
import sys
import subprocess

def train():
    # Run the train.py script
    subprocess.run(["python", "train.py"])

def evaluate():
    # Run the eval.py script
    subprocess.run(["python", "eval.py"])

if __name__ == "__main__":
    # Run training
    train()

    # Run evaluation
    evaluate()
