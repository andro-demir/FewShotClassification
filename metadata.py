'''
MiniImagenet(100 classes): 
Derived from ILSVRC-2012
Consists of 60000, 84x84 RGB images with 600 images per class.
These classes are randomly split into 64, 16 and 20 classes for
meta-training, meta-validation and meta-testing
To perform meta-validation and meta-test on unseen and classes, we isolate
16 and 20 classes from the original set of 100, 
leaving 64 classes for the training tasks.

TieredImagenet:
Derived from ILSVRC-2012
608 classes divided into 351 meta-training classes, 97 meta-validation classes 
and 160 meta-test classes. 
Consists of 779165 RGB images with size 84×84

CIFAR-FS(100 classes):
Derived from CIFAR-100
Consists of 60000, 32x32 RGB images with 600 images per class.
These classes are randomly split into 64, 16 and 20 classes for
meta-training, meta-validation and meta-testing

FC100:
Based on CIFAR-100 with the objective to minimize the information overlap 
between class splits. 32x32 color images belonging to 100 different classes
are further grouped into 20 superclasses. We split the dataset by
superclass, rather than by individual class to minimize the information overlap 
Thus the train split contains 60 classes belonging to 12 superclasses, 
the validation and test contain 20 classes belonging to 5 superclasses each.

Each dataset D_i in the meta-train, meta-validation and meta-test data 
is separated in two parts: a training set (with N unique data) to adapt  
the model to the task at hand, and a test set (or query set) 
for evaluation and meta-optimization. 
No same image can be both in the training and test set. 

Training policy:
Given N novel classes with only K examples each in train_inputs, 
predict queries in test_inputs to one of N classes 
'''

import numpy as np
from torchmeta.datasets import MiniImagenet, TieredImagenet, CIFARFS, FC100, Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

import matplotlib.pyplot as plt

class Metadata_generator():
    def __init__(self, N, K, num_test_per_class, batch_size, dataset):
        '''
        N-way K-shot training
        N: number of different unseen classes 
        K: how many examples per class for evaluation
        '''
        self.N = N
        self.K = K
        self.num_test_per_class = num_test_per_class
        self.batch_size = batch_size
        self.dataset = dataset

    def generate_batch(self):
        '''
        The data-loaders of torch meta are fully compatible with standard data
        components of PyTorch, such as Dataset and DataLoade+r.
        Augments the pool of class candidates with variants, such as rotated images
        ''' 
        if self.dataset == "miniImageNet":
            dataset = MiniImagenet("data",
                            # Number of ways
                            num_classes_per_task=self.N,
                            # Resize the images and converts them 
                            # to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(84), ToTensor()]),
                            # Transform the labels to integers 
                            target_transform=Categorical(num_classes=self.N),
                            # Creates new virtual classes with rotated versions
                            # of the images (from Santoro et al., 2016)
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_train=True,
                            download=True)

        if self.dataset == "tieredImageNet":
            dataset = TieredImagenet("data",
                            # Number of ways
                            num_classes_per_task=self.N,
                            # Resize the images and converts them 
                            # to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(32), ToTensor()]),
                            # Transform the labels to integers 
                            target_transform=Categorical(num_classes=self.N),
                            # Creates new virtual classes with rotated versions
                            # of the images (from Santoro et al., 2016)
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_train=True,
                            download=True) 

        if self.dataset == "CIFARFS":
            dataset = CIFARFS("data",
                            # Number of ways
                            num_classes_per_task=self.N,
                            # Resize the images and converts them 
                            # to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(84), ToTensor()]),
                            # Transform the labels to integers 
                            target_transform=Categorical(num_classes=self.N),
                            # Creates new virtual classes with rotated versions
                            # of the images (from Santoro et al., 2016)
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_train=True,
                            download=True)

        if self.dataset == "FC100":
            dataset = FC100("data",
                            # Number of ways
                            num_classes_per_task=self.N,
                            # Resize the images and converts them 
                            # to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(32), ToTensor()]),
                            # Transform the labels to integers 
                            target_transform=Categorical(num_classes=self.N),
                            # Creates new virtual classes with rotated versions
                            # of the images (from Santoro et al., 2016)
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_train=True,
                            download=True) 

        if self.dataset == "Omniglot":
            dataset = Omniglot("data",
                            # Number of ways
                            num_classes_per_task=self.N,
                            # Resize the images and converts them 
                            # to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(28), ToTensor()]),
                            # Transform the labels to integers 
                            target_transform=Categorical(num_classes=self.N),
                            # Creates new virtual classes with rotated versions
                            # of the images (from Santoro et al., 2016)
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_train=True,
                            download=True)      
                        
        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=self.K, 
                                num_test_per_class=self.num_test_per_class)

        dataloader = BatchMetaDataLoader(dataset, batch_size=self.batch_size,
                                         num_workers=2)
        return dataloader


def test_loading(train_inputs, train_targets, 
                 test_inputs, test_targets, K, num_test_per_class, eps=0):
    '''
    Visualize train inputs and targets for one episode in the batch
    eps(int): episode number in a batch
    '''
    def imshow(img):
        '''
        img is the tensor
        '''
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    episode_trainX, episode_trainY = train_inputs[eps], train_targets[eps]
    episode_testX, episode_testY = test_inputs[eps], test_targets[eps]

    # show images, each row for a different label
    imshow(torchvision.utils.make_grid(episode_trainX, nrow=K))
    imshow(torchvision.utils.make_grid(episode_testX, nrow=num_test_per_class))
    #exit(1)
    

def main():
    dataset = "miniImageNet"
    # dataset = "tieredImageNet"
    # dataset = "CIFARFS"
    # dataset = "FC100"
    # dataset = "Omniglot"
    N_way = 5
    K_shot = 5
    num_test_per_class = 15
    batch_size = 16 # number of episodes
    data_generator = Metadata_generator(N_way, K_shot, num_test_per_class,
                                        batch_size, dataset)
    dataloader = data_generator.generate_batch()
    for idx, batch in enumerate(dataloader):
        print("Batch id %i" %idx)
        train_inputs, train_targets = batch["train"]
        # train inputs: [batch_size, N_way*K_shot, 3, img_size, img_size]
        print('Train inputs shape: {0}'.format(train_inputs.shape))   
        # train targets: [batch_size, N_way*K_shot] 
        print('Train targets shape: {0}'.format(train_targets.shape))

        test_inputs, test_targets = batch["test"]
        # test_inputs: [batch_size, N_way*num_test_per_class, 3, img_size, img_size]
        print('Test inputs shape: {0}'.format(test_inputs.shape))  
        # test_outputs: [batch_size, N_way*num_test_per_class]    
        print('Test targets shape: {0}'.format(test_targets.shape))    

        test_loading(train_inputs, train_targets, 
                     test_inputs, test_targets, K_shot, num_test_per_class)  


if __name__ == '__main__':
    main()