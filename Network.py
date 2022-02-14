from audioop import add
from scipy.fftpack import sc_diff
import torch
from torch import nn

class Network(nn.Sequential):
    def __init__(self, dim_in = 28, dim_out = 10):
        super().__init__()
        self.dict = {
            "Linear" : 1,
            "Relu" : 1,
            "Elu" : 1, 
            "Tanh" : 1,
            "MaxPool" : 1,
            "AvgPool" : 1,
            "Conv2d" : 1,
            "Sigmoid" : 1,
            "SoftMax" : 1,
            "LogSigmoid" : 1,
            "LogSoftMax" : 1,
            "Dropout" : 1
        }
        self.net_param = {
            "pic_size" : dim_in,
            "chanel" : 1,
            "can_add_conv" : True 
        }
        self.dim_in = dim_in
        self.dim_out = dim_out
       
    
    def addLinearLayer(self, dim_out, dropout = 0.1, bias = True):
        assert dropout >= 0. and dropout <= 0.6
        assert dim_out >= 10
        self.__addLinear(dim_out, dropout = dropout, bias=bias)

    def addActiveLayer(self, type = 0):
        assert type in [0, 1, 2]
        if type == 0:
            self.__addRelu()
        elif type == 1:
            self.__addElu()
        else:
            self.__addTanh()

    def addLastLayer(self, type = 0):
        assert type in [0, 1, 2, 3]
        if type == 0:
            self.__addSigmoid()
        elif type == 1:
            self.__addLogSigmoid()
        elif type == 2:
            self.__addSoftMax()
        else:
            self.__addLogSoftMax()

    def addConvAndPool(self, type, type_active, out_channels, kernel_size = 0, padding = 0, bias = True):
        # print(f"type is {type}")
        assert type in [0, 1]
        assert self.net_param["can_add_conv"] == True and self.net_param["pic_size"] >= 8 and self.net_param["pic_size"] % 2 == 0
        self.__addConv2d(out_channels, kernel_size, padding, bias)
        self.addActiveLayer(type_active)
        if type == 0:
            self.__addMaxPool()
        else:
            self.__addAvgPool()

    def __addConv2d(self, out_channels, kernel_size = 0, padding = 0, bias = True):
        assert kernel_size in [0, 1]
        if kernel_size == 0:
            kernel_size = 3
        else:
            kernel_size = 5
        assert padding in [0, 1, 2]
        # （in_size - K + 2P）/ S +1
        index = self.dict["Conv2d"]
        name = f"Conv2d_{index}"
        in_channels = self.net_param["chanel"]
        self.add_module(name, nn.Conv2d(
            in_channels=in_channels,              
            out_channels=out_channels,            
            kernel_size=kernel_size,             
            stride=1,                   
            padding=padding, 
            bias=bias 
        ))
        self.dict["Conv2d"] = index + 1
        self.net_param["chanel"] = out_channels
        pic_size = self.net_param["pic_size"]
        self.net_param["pic_size"] = (pic_size - kernel_size + 2 * padding) + 1
    
    def __addMaxPool(self):
        index = self.dict["MaxPool"]
        name = f"MaxPool_{index}"
        self.add_module(name, nn.MaxPool2d(kernel_size=2))
        self.dict["MaxPool"] = index + 1
        pic_size = self.net_param["pic_size"]
        self.net_param["pic_size"] = pic_size // 2

    def __addAvgPool(self):
        index = self.dict["AvgPool"]
        name = f"AvgPool_{index}"
        self.add_module(name, nn.AvgPool2d(kernel_size=2))
        self.dict["AvgPool"] = index + 1
        pic_size = self.net_param["pic_size"]
        self.net_param["pic_size"] = pic_size // 2

    def __addLinear(self, dim_out, dropout = 0.1, bias = True):
        index = self.dict["Linear"]
        name = f"Linear_{index}"
        if self.net_param["can_add_conv"]:
            dim = self.net_param["pic_size"]**2 * self.net_param["chanel"]
        else:
            dim = self.net_param["pic_size"]
        self.add_module(name, nn.Linear(dim, dim_out, bias = bias))
        dropout_index = self.dict["Dropout"]
        drop_name = f"Dropout_{dropout_index}"
        self.add_module(drop_name, nn.Dropout(p = dropout))
        self.dict["Linear"] = index + 1
        self.dict["Dropout"] = dropout_index + 1
        self.net_param["pic_size"] = dim_out
        self.net_param["chanel"] = 1
        self.net_param["can_add_conv"] = False


    def __addRelu(self):
        index = self.dict["Relu"]
        name = f"Relu_{index}"
        self.add_module(name, nn.ReLU())
        self.dict["Relu"] = index + 1
    
    def __addElu(self):
        index = self.dict["Elu"]
        name = f"Elu_{index}"
        self.add_module(name, nn.ELU())
        self.dict["Elu"] = index + 1

    def __addTanh(self):
        index = self.dict["Tanh"]
        name = f"Tanh_{index}"
        self.add_module(name, nn.Tanh())
        self.dict["Tanh"] = index + 1

    def __addSigmoid(self):
        index = self.dict["Sigmoid"]
        name = f"Sigmoid_{index}"
        self.add_module(name, nn.Sigmoid())
        self.dict["Sigmoid"] = index + 1
    
    def __addLogSigmoid(self):
        index = self.dict["LogSigmoid"]
        name = f"LogSigmoid_{index}"
        self.add_module(name, nn.LogSigmoid())
        self.dict["LogSigmoid"] = index + 1
    
    def __addSoftMax(self):
        index = self.dict["SoftMax"]
        name = f"SoftMax{index}"
        self.add_module(name, nn.Softmax())
        self.dict["SoftMax"] = index + 1

    def __addLogSoftMax(self):
        index = self.dict["LogSoftMax"]
        name = f"LogSoftMax_{index}"
        self.add_module(name, nn.LogSoftmax())
        self.dict["LogSoftMax"] = index + 1

    