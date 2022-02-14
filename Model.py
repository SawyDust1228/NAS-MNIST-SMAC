import torch
from torch import nn, optim
from Network import Network

class Model(nn.Module):
    def __init__(self,type_pool = 1, type_active = 0, out_channels= 10, 
        kernel_size=0, padding=0, linear_layer_out_1 = 100, active_type_1 = 1,
        linear_layer_out_2 = 50, active_type_2 = 2,
        linear_layer_out_3 = 10, last_layer_type = 2, drop_out_1 = 0.1, drop_out_2 = 0.1, drop_out_3 = 0.1) -> None:
        super().__init__()
        self.network = Network()
        # type, type_active, out_channels, kernel_size = 3, padding = 0, bias = True
        self.network.addConvAndPool(type = type_pool, type_active = type_active, out_channels= out_channels, kernel_size=kernel_size, padding=padding)
        # self.network.addConvAndPool(0 ,1, 10)
        self.network.addLinearLayer(linear_layer_out_1, dropout = drop_out_1)
        self.network.addActiveLayer(active_type_1)
        self.network.addLinearLayer(linear_layer_out_2, dropout = drop_out_2)
        self.network.addActiveLayer(active_type_2)
        self.network.addLinearLayer(linear_layer_out_3, dropout = drop_out_3)
        self.network.addLastLayer(last_layer_type)

    def forward(self, x):
        for net in self.network:
            if isinstance(net, nn.Linear):
                # shape = x.shape
                batch = x.shape[0]
                x = x.view(batch, -1)
            x = net(x)
        return x
