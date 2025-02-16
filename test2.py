import torch
import torch_directml

device = torch_directml.device()  # Use DirectML explicitly
print("Using device:", device)
