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
superclass, rather than by individual class to minimize the information overlap. 
Thus the train split contains 60 classes belonging to 12 superclasses, 
the validation and test contain 20 classes belonging to 5 superclasses each.
'''

from torchmeta.datasets import MiniImagenet, CIFARFS, FC100, TieredImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchmeta.utils.data import BatchMetaDataLoader
import torchvision.transforms as transforms

class Metadata():
    def __init__(self, K, )
dataset = Omniglot("data",
                   # Number of ways
                   num_classes_per_task=5,
                   # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                   transform=Compose([Resize(28), ToTensor()]),
                   # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                   target_transform=Categorical(num_classes=5),
                   # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_train=True,
                   download=True)
                   
dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=5, num_test_per_class=15)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

def main():
    dataloader = load_data(dataset)

    for batch in dataloader:
        train_inputs, train_targets = batch["train"]
        print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
        print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

        test_inputs, test_targets = batch["test"]
        print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
        print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75) 


if __name__ == '__main__':
    main()