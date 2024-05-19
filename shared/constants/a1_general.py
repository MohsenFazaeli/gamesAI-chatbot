import torch
# general app constans
stage = "dev"  # "" for prod and "dev" for development

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

seed_data_size = 260