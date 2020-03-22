import torch
from transformers import AlbertForMaskedLM

PATH = "./albert-base-v2"
model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
torch.save(model.state_dict(), PATH)
