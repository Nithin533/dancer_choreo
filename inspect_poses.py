import numpy as np
import os

for root, dirs, files in os.walk("output"):
    for f in files:
        if "poses_smooth" in f:
            path = os.path.join(root, f)
            data = np.load(path)
            print(f"File  : {path}")
            print(f"Shape : {data.shape}")
            print(f"Min   : {data.min():.4f}")
            print(f"Max   : {data.max():.4f}")
            print(f"Mean  : {data.mean():.4f}")
            print(f"Std   : {data.std():.4f}")
            print(f"Frame 0 first 6 values: {data[0][:6]}")
            print()
