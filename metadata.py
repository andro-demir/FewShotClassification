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
'''

from torchmeta.datasets import MiniImagenet, CIFARFS, FC100, TieredImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

class Metadata_generator():
    def __init__(self, K, N, num_test_per_class, batch_size, dataset):
        '''
        K-shot N-way learning, each N from different unseen classes 
        '''
        self.K = K
        self.N = N
        self.num_test_per_class = num_test_per_class
        self.batch_size = batch_size
        self.dataset = dataset

    def generate_batch(self):
        '''
        The data-loaders of torch meta are fully compatible with standard data
        components of PyTorch, such as Dataset and DataLoader.
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
                        
        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=self.K, 
                                num_test_per_class=self.num_test_per_class)

        dataloader = BatchMetaDataLoader(self.dataset, batch_size=self.batch_size,
                                         num_workers=2)
        return dataloader

def main():
    dataset = "miniImageNet"
    # dataset = "CIFARFS"
    # dataset = "FC100"
    # dataset = "tieredImageNet"
    data_generator = Metadata_generator(5, 5, 15, 16, dataset)
    dataloader = data_generator.generate_batch()
    for batch in dataloader:
        train_inputs, train_targets = batch["train"]
        print('Train inputs shape: {0}'.format(train_inputs.shape))    
        print('Train targets shape: {0}'.format(train_targets.shape))  

        test_inputs, test_targets = batch["test"]
        print('Test inputs shape: {0}'.format(test_inputs.shape))      
        print('Test targets shape: {0}'.format(test_targets.shape))    


if __name__ == '__main__':
    main()