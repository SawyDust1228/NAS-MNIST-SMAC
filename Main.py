from Classifier import Classifier
import os
import torch

seed = 1234567890
torch.manual_seed(seed=seed)

if __name__ == "__main__":
    os.system('export DISPLAY=:1')
    classifier = Classifier(epoch=20)
    classifier.printNet()
    # classifier.plotImage()
    classifier.train()
    classifier.test() 
    print(f"[ACCURACY_LIST]: {classifier.accuracy_list}")