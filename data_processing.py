import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# Image class comes from a package called pillow 
# PIL used as the format for passing images into torchvision

batch_size = 64

def chained_transformation():
    '''
    Train set is cropped randomly, flipped horizontally and normalized.
    Test set is normalized.
    RandomCrop extracts a patch of size (32, 32) from the input image randomly
    RandomHorizontalFlip flips the cropped image horizontally 
    for data augmentation
    ToTensor converts the input image to torch tensor
    Do not change the mean and standard deviation values for normalize
    '''
    preprocess = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010))])
    return preprocess

def normalize_testset():
    '''
    For test set, we only normalize the dataset
    Wothout data augmentation.
    '''
    normalized = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010))])
    return normalized

def load_cifar10():
    '''
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
    with 6000 images per class. 
    There are 50000 training images and 10000 test images.
    '''
    preprocess = chained_transformation()
    normalized = normalize_testset()
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                            download=True, 
                                            transform=preprocess)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                           download=True, 
                                           transform=normalized)
    return trainset, testset

def load_test_image(image_path):
    image = Image.open(image_path)
    image.show()
    image = image.resize((32, 32))
    preprocess = normalize_testset()
    img_tensor = preprocess(image)
    return img_tensor

def generate_batches(trainset, testset, batch_size=batch_size):
    '''
    batch_size(int): number of samples contained in each generated batch.
    shuffles the order in which examples are fed to the classifier
    so that batches between epochs do not look alike
    num_workers is the number of processes that generate batches in parallel. 
    A high enough num_workers assures that CPU computations are 
    efficiently managed
    '''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def test_loading(trainloader, classes):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    # show images
    imshow(torchvision.utils.make_grid(images))

def main():
    trainset, testset = load_cifar10()
    trainloader, testloader = generate_batches(trainset, testset)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    test_loading(trainloader, classes)


if __name__ == '__main__':
    main()