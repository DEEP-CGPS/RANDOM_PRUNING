import os
from simplify.fuse import fuse
import torch
import torch.nn as nn
import simplify
from torchsummary import summary
import torch.nn.utils.prune as prune


def prune_model(model, args):
    if not os.path.exists("models"):
            os.makedirs("models")
    torch.manual_seed(args.seed)
    pos = 0
    model.to(args.device)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if args.method == 'random':
                prune.random_structured(module, 'weight', amount=args.list_pruning[pos], dim=0)
            elif args.method == 'weight':
                prune.ln_structured(module, 'weight', amount=args.list_pruning[pos],dim=0,n=2)
            prune.remove(module, 'weight')
            pos+=1
        if isinstance(module, nn.Linear):
            if args.method == 'random':
                prune.random_structured(module, 'weight', amount=args.list_pruning[pos], dim=0)
            elif args.method == 'weight':
                prune.ln_structured(module, 'weight', amount=args.list_pruning[pos],dim=0,n=2)
            prune.remove(module,'weight')
            pos+=1

    simplify.simplify(model, torch.ones((1, 3, 224, 224)).to(args.device), fuse_bn=False)

    torch.save(model,f'models/{args.model_architecture}_{args.dataset}_{args.method}_{args.model_type}.pth')