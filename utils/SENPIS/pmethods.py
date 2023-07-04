import torch.nn.utils.prune as prune
import torch
from . import auxiliarFC
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import operator
import numpy as np
import torch.nn as nn

BatchSampler=auxiliarFC.BatchSampler

class MaskSimilitudev1(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    def __init__(self,listpruning):
        self.listpruning=listpruning
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[self.listpruning]=0
        return mask
    
class SenpisFaster():
    
  def __init__(self, Net, n_classes, dataset = None, amount = None, only_conv = False, n_samples = 5, sigma = 1, attenuation_coefficient = 0.9 ):
      self.Net=Net
      self.amount=amount
      self.dataset=dataset
      self.n_classes= n_classes
      self.n_samples= n_samples
      self.sigma= sigma
      self.attenuation_coefficient= attenuation_coefficient
      self.only_conv = only_conv
      self.MakeBatch()
      self.run()

  def run(self):

    ActualConv=1
    ActualFc=1

    with torch.no_grad():
      L=0
      for name, module in self.Net.named_modules():
          if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)) and self.only_conv == False:
            L+=1
          elif isinstance(module, torch.nn.Conv2d) and self.only_conv == True:
            L+=1
      if self.only_conv == False:
        data = 1
      else:
        data = 0
      cont=0
      layers_to_prune = L-data

      if isinstance(self.amount, list) and len(self.amount) != 2:
        list_prune = np.linspace(self.amount[0], self.amount[1], num = layers_to_prune)
        print("list_prune :",list_prune)
      if isinstance(self.amount, list) and len(self.amount) > 2:
        list_prune = self.amount
        print("list_prune :",list_prune)
      for name, module in self.Net.named_modules():
        if  isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) and cont < layers_to_prune:
            if isinstance(module,torch.nn.Conv2d):
              self.val=1
              a="Conv"+str(ActualConv)
              ActualConv+=1
            else:
              self.val=0
              a="Fc"+str(ActualFc)
              ActualFc+=1

            if isinstance(self.amount, list):
              pf = list_prune[cont]
            else:
              pf = self.amount

            self.module=module
            
            self.loss=self.ClassLoss()
            self.fm = self.module.bias[:].size()[0]

            self.IM()
            
            numpf=round(pf*self.fm)
            sorted, indices = torch.sort(self.IM_Global)
            self.listpruning=indices[0:numpf]
            MaskSimilitudev1.apply(self.module, name='weight',listpruning=self.listpruning)
            prune.remove(self.module, 'weight')
            MaskSimilitudev1.apply(self.module, name='bias',listpruning=self.listpruning)
            prune.remove(self.module, 'bias')
            print(f'layer {cont} pruned')
            cont+=1

  def IM(self):
    with torch.no_grad():
      self.IM_Local=torch.empty((self.n_classes,self.fm))

      for n in range(self.fm):      
        self.weight_backup=self.module.weight[n].clone()
        self.bias_backup=self.module.bias[n].clone()
        self.module.weight[n]*=0
        self.module.bias[n]*=0  
        self.actual_loss=self.ClassLoss()
        self.IM_Local[:,n]=abs(self.actual_loss-self.loss)
        self.module.weight[n]=self.weight_backup.clone()
        self.module.bias[n]=self.bias_backup.clone()

      self.IM_Global=self.IM_Local.clone()
      self.IM_Global=torch.mean(self.IM_Global,dim=0)


  def ClassLoss(self):
    n_samples = self.n_samples 
    L = nn.CrossEntropyLoss()
    with torch.no_grad():
      loss=torch.empty(self.n_classes)
      output=self.Net(self.x)
      for i in range(self.n_classes):
        out = output[i*n_samples:i*n_samples+n_samples,:].to(self.device)
        lab = self.labels[i*n_samples:i*n_samples+n_samples].to(self.device)
        loss[i]=L(out,lab)
        
    return loss

  def MakeBatch(self):
      
      self.device=next(self.Net.parameters()).device
      bbatch_sampler = BatchSampler(self.dataset,
                                    self.n_classes, self.n_samples)
      dataloader = torch.utils.data.DataLoader(self.dataset,
                                                batch_sampler=bbatch_sampler)
      batchp= iter(dataloader)
      self.x, self.labels = batchp.__next__()
      self.x=self.x.to(self.device)
  