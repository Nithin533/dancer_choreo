import torch

print("STARTED")

path = "best_model/data.pkl"

print("Loading...")

obj = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

print("LOADED")
print(type(obj))

if isinstance(obj, dict):
    print("Keys:")
    print(obj.keys())

print("DONE")