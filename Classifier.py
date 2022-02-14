from torch import device, nn, optim
import torch
from Model import Model
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Classifier():
    def __init__(self, epoch, type_pool = 1, type_active = 0, out_channels= 10, 
        kernel_size=0, padding=0, linear_layer_out_1 = 100, active_type_1 = 1,
        linear_layer_out_2 = 50, active_type_2 = 2,
        linear_layer_out_3 = 10, last_layer_type = 3, drop_out_1 = 0.1, drop_out_2 = 0.1, drop_out_3 = 0.1,
        optim_type = 0, lr = 10e-2) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__addLoss()
        self.__loadData()
        self.accuracy_list = []
        self.epoch = epoch


        self.addModel(type_pool = type_pool, type_active = type_active, out_channels= out_channels, 
        kernel_size=kernel_size, padding=padding, linear_layer_out_1 = linear_layer_out_1, active_type_1 = active_type_1,
        linear_layer_out_2 = linear_layer_out_2, active_type_2 = active_type_2,
        linear_layer_out_3 = linear_layer_out_3, last_layer_type = last_layer_type, optim_type = optim_type, lr = lr, 
        drop_out_1 =drop_out_1, drop_out_2 = drop_out_2, drop_out_3 = drop_out_3)

        

    def addModel(self, type_pool = 1, type_active = 0, out_channels= 10, 
        kernel_size=0, padding=0, linear_layer_out_1 = 100, active_type_1 = 1,
        linear_layer_out_2 = 50, active_type_2 = 2,
        linear_layer_out_3 = 10, last_layer_type = 3, 
        optim_type = 0, lr = 10e-2, drop_out_1 = 0.1, drop_out_2 = 0.1, drop_out_3 = 0.1):

        # kernel_size = self.kernelSizeToInteger(kernel_size)
        # 定义Model
        self.model = Model(type_pool = type_pool, type_active = type_active, out_channels= out_channels, 
        kernel_size=kernel_size, padding=padding, linear_layer_out_1 = linear_layer_out_1, active_type_1 = active_type_1,
        linear_layer_out_2 = linear_layer_out_2, active_type_2 = active_type_2,
        linear_layer_out_3 = linear_layer_out_3, last_layer_type = last_layer_type, 
        drop_out_1 = drop_out_1, drop_out_2 = drop_out_2, drop_out_3 = drop_out_3)
        # 定义学习lv
        self.lr = lr
        # 定义优化器
        self.__addOptimizer(optim_type)
        # GPU
        self.model = self.model.to(self.device)

    def kernelSizeToInteger(self, kernel_size):
        return int(kernel_size)

    def __addOptimizer(self, type = 0):
        assert type in [0, 1]
        if type == 0:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def __addLoss(self):
        self.criterion = nn.CrossEntropyLoss()

    def printNet(self):
        print(self.model)

    def plotImage(self):
        plt.figure(figsize=(16, 6))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            image, _ = self.train_loader.dataset.__getitem__(i)
            plt.imshow(image.squeeze().numpy())
            plt.axis('off')

    def __loadData(self):
        self.train_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
                        batch_size=64, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data', train=False, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
                        batch_size=1000, shuffle=True)

    def train(self, perm=torch.arange(0, 784).long()):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # send to device
            data, target = data.to(self.device), target.to(self.device)
            
            # permute pixels
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)

            self.optimizer.zero_grad()
            output = self.model(data)

            # print(f"[OUTPUT]: {output.shape}, [TARGET]: {target.shape}")


            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                
    def test(self, perm=torch.arange(0, 784).long()):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            # send to device
            data, target = data.to(self.device), target.to(self.device)
            
            # permute pixels
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            output = self.model(data)
            test_loss += self.criterion(output, target).item() # sum up batch loss                                                               
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            accuracy))

