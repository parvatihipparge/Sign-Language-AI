# create_dummy_model.py
import torch
from model.model import LandmarkClassifier

model = LandmarkClassifier(input_dim=63, hidden_dim=128, num_classes=5)
torch.save({"model_state": model.state_dict()}, "model/checkpoint.pth")
print("Dummy model saved at model/checkpoint.pth")
