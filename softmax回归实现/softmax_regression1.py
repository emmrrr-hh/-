from random import shuffle

import torch
import torchvision
from d2l.mxnet import download, get_dataloader_workers
#from d2l.mxnet import get_fashion_mnist_labels
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

#下载图片训练集和测试集
trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)

# print(len(mnist_train),'\n',len(mnist_test))
#
# print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle shoot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # x轴隐藏
        ax.axes.get_yaxis().set_visible(False)  # y轴隐藏
        if titles:
            ax.set_title(titles[i])  # 显示标题
    return axes


X,y=next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))


def get_data_loader_workers():
    """定义读取数据的进程个数"""
    return 4


def load_data_fashion_minst(batch_size,resize=None):
    trans=[transform.Totensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)(
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=true,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,tranform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_data_loader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=get_dataloader_workers()))