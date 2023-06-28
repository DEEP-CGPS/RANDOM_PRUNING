from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
import torch
from torch import nn

class ModelParams():
    def __init__(self,model,model_input)->object:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model_input = model_input.to(device)
        
    def get_flops(self)->int:
        flops = FlopCountAnalysis(self.model, self.model_input)
        return flops.total()
    
    def get_summary(self):
        print(summary(self.model, tuple(self.model_input.size())))
        
    def get_times_layer(self,layer_type:str="Conv2d")->int:
        count_layers = 0      
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and layer_type == "Conv2d":
                count_layers += 1
            if isinstance(module, nn.Linear) and layer_type == "Linear":
                count_layers += 1
        return count_layers           
        
    def get_all_params(self):
        flops = self.get_flops()*2
        self.get_summary()
        conv_layers = self.get_times_layer(layer_type="Conv2d")
        linear_layers = self.get_times_layer(layer_type="Linear")
        
        return flops, conv_layers, linear_layers