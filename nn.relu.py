import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)

class Adding(nn.Module):
    def __init__(self):
        super(Adding,self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()
    def forward(self,input):
        output=self.sigmoid1(input)
        return output

adding=Adding()

writer=SummaryWriter("./logs_relu")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=adding(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()