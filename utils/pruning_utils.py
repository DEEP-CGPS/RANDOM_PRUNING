import os
from simplify.fuse import fuse
import torch
import torch.nn as nn
import simplify
from torchsummary import summary
import torch.nn.utils.prune as prune
torch.manual_seed(23)

def prune_model(model, args):
    if not os.path.exists("models"):
            os.makedirs("models")
    pos = 0
    model.to(args.device)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=args.list_pruning[pos], dim=0)
            prune.remove(module, 'weight')
            pos+=1
        if isinstance(module, nn.Linear):
            prune.random_structured(module, 'weight', amount=args.list_pruning[pos], dim=0)
            prune.remove(module, 'weight')
            pos+=1

    simplify.simplify(model, torch.ones((1, 3, 224, 224)).to(args.device), fuse_bn=False)

    torch.save(model,f'models/{args.pruned_model_name}.pth')