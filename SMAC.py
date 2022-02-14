from skopt import dummy_minimize, forest_minimize, gp_minimize
import skopt
from skopt.space import Real,Integer,Categorical
from skopt.utils import use_named_args
import random
import torch

seed = 1234567890
torch.manual_seed(seed=seed)

from Classifier import Classifier
epoch = 20
space = [
    Integer(0, 1, name = "type_pool"),
    Integer(0, 2, name = "type_active"),
    Integer(5, 16, name = "out_channels"),
    Integer(0, 1, name = "kernel_size"),
    Integer(0, 2, name = "padding"),
    Integer(100, 200, name = "linear_layer_out_1"),
    Integer(0, 2, name = "active_type_1"),
    Integer(20, 100, name = "linear_layer_out_2"),
    Integer(0, 2, name = "active_type_2"),
    Integer(0, 3, name = "last_layer_type"),
    Integer(0, 1, name = "optim_type"),
    Real(10e-4, 10e-1, "log-uniform", name="lr"),
    Real(0.0, 0.6, name = "drop_out_1"),
    Real(0.0, 0.6, name = "drop_out_2"),
    Real(0.0, 0.6, name = "drop_out_3")
]

classifier = Classifier(epoch=epoch)
param_list = []
@use_named_args(space)
def objective(**param):
    # print(f"[DEBUG]: {param}")
    param_list.append(param)
    classifier.addModel(**param)
    classifier.train()
    classifier.test() 
    l =  classifier.accuracy_list
    return -l[len(l) - 1]


# result = forest_minimize(objective, space, random_state=0, n_calls=100)
# result = dummy_minimize(objective, space, random_state=0, n_calls=100)
result = gp_minimize(objective, space, random_state=0, n_calls=100)
print(f"[RESULT]: {classifier.accuracy_list}")
print(f"[FINAL PARAMETER]: {param_list[len(param_list) - 1]}")
print(classifier.printNet())

with open("log_gp", "w") as f:
    f.write(f"[RESULT]: {classifier.accuracy_list}" + "\n")
    f.write(f"[FINAL PARAMETER]: {param_list[len(param_list) - 1]}" + "\n")
    
