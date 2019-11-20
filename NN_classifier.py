'''
TO DO:
1. Do not force the network to make a prediction for every input.
   If there's uncertainty, let it say I don't know
2. creating a dictionary and training on cosine similarity scores 
'''

import numpy as np
import fire
import data_processing as dp
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from metadata import Metadata_generator as MG
from utils.helper import save_model, load_model, set_device
import matplotlib.pyplot as plt

# dataset = "cifar10"
# dataset = "mnist"
dataset = "miniImageNet"
# dataset = "tieredImageNet"
# dataset = "CIFARFS"
# dataset = "FC100"
# dataset = "Omniglot"

if dataset not in ["cifar10", "mnist"]:
    N_way = 5
    K_shot = 5
    num_test_per_class = 15
    batch_size = 16 # number of episodes
    data_generator = MG(N_way, K_shot, num_test_per_class,
                        batch_size, dataset)

class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        '''
        in_planes(int): number of input channels
        out_planes(int): number of output channels
        stride(int): controls the stride for cross-correlation
        padding is 0 by default in convolutions
        
        batch norm calculates the mean and standard deviation per-dimension 
        over the minibatches during training. At test time, it takes the mean 
        and variance estimated during training. 
        By the strong law of large numbers, this should give a good estimation
        '''
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, 
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion*out_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*out_planes, kernel_size=1, 
                          stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*out_planes)
            )

    def forward(self, x):
        '''
        Note: softPlus activation function is a smooth approximation to ReLU
        it constrains the output to always be positive.
        '''
        out = F.softplus(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.softplus(out)
        return out


class resnet(nn.Module):
    '''
    Defines the network architecture, activations and regularizers.
    Forward prop.
    First convolutional layer has kernel size 5x5, stride 1 and the 
    total number of kernels is 32 
    '''
    def __init__(self, dataset, block, num_blocks, num_classes):
        super(resnet, self).__init__()
        self.in_planes = 32
        in_ch = 1 if dataset == "mnist" or dataset == "Omniglot" else 3
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=5, 
                               stride=1, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.layer1 = self.make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 256, num_blocks[3], stride=2)  
        if dataset in ["mnist", "cifar10", "Omniglot", "CIFARFS", "FC100"]:
            W = 256
        if dataset in ["miniImageNet", "tieredImageNet"]:
            W = 1024
        self.dense = nn.Linear(W * block.expansion, num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.softplus(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.avg_pool2d(x, kernel_size=4)
        # view reshape tensor x. we don't know how many columns we want
        # but are sure of the number of rows, so we specify this with a -1
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


def ResNet18(dataset=dataset):
    if dataset in ["cifar10", "mnist"]:
        num_classes = 10
    else:
        num_classes = N_way
    return resnet(dataset, basic_block, [2,2,2,2], num_classes).cuda()

def probabilistic_model(inputs, labels):
    '''
    pyro.random_module() converts weights and biases into random variables 
    that have the prior probability distribution given by 
    dense_weight_prior and dense_bias_prior for a normal distribution
    this "overloads" the parameters of the random module with
    samples from the prior!
    '''
    resnet = ResNet18()
    dense_weight_prior = Normal(loc=torch.zeros_like(resnet.dense.weight), 
                                scale=torch.ones_like(resnet.dense.weight))
    dense_bias_prior = Normal(loc=torch.zeros_like(resnet.dense.bias), 
                              scale=torch.ones_like(resnet.dense.bias))
       
    priors = {'dense.weight': dense_weight_prior, 
              'dense.bias': dense_bias_prior}
    
    lifted_module = pyro.random_module("module", resnet, priors)
    
    # This samples a neural network (which also samples weights and biases)
    # we wrap the nn model with random_module and sample and instance
    # of the nn
    sampled_nn_model = lifted_module()
    
    # runs the sampled nn on the input data
    lhat = F.log_softmax(sampled_nn_model(inputs))
    
    # this shows the output of the network will be categorical
    pyro.sample("obs", Categorical(logits=lhat), obs=labels)

def probabilistic_guide(inputs, labels):
    resnet = ResNet18()
    # Dense layer weight distribution priors
    dense_w_mu = torch.randn_like(resnet.dense.weight)
    dense_w_sigma = torch.randn_like(resnet.dense.weight)
    dense_w_mu_param = pyro.param("dense_w_mu", dense_w_mu)
    dense_w_sigma_param = F.softplus(pyro.param("dense_w_sigma", dense_w_sigma))
    dense_w_prior = Normal(loc=dense_w_mu_param, scale=dense_w_sigma_param)
    
    # Dense layer bias distribution priors
    dense_b_mu = torch.randn_like(resnet.dense.bias)
    dense_b_sigma = torch.randn_like(resnet.dense.bias)
    dense_b_mu_param = pyro.param("dense_b_mu", dense_b_mu)
    dense_b_sigma_param = F.softplus(pyro.param("dense_b_sigma", dense_b_sigma))
    dense_b_prior = Normal(loc=dense_b_mu_param, scale=dense_b_sigma_param)
    
    priors = {'dense.weight': dense_w_prior, 
              'dense.bias': dense_b_prior}
    
    lifted_module = pyro.random_module("module", resnet, priors)
    
    return lifted_module()

def inference(scheduler=True):
    '''
    Sets the loss and optimization criteria and number of epochs.
    scheduler decays the learn rate by a multiplicative factor of 
    gamma at each epoch.
    They were chosen heuristically.
    To do inference use stochastic variational inference (SVI)
    at each iteration of the training loop it will take a gradient step 
    with respect to the ELBO objective
    '''
    if scheduler == True:
        optimizer = pyro.optim.ExponentialLR({'optimizer': optim.Adam, 
                                              'optim_args': {'lr': 0.01}, 
                                              'gamma': 0.95})
    else:
        optimizer = pyro.optim.Adam({"lr": 0.01})
    
    svi = SVI(probabilistic_model, probabilistic_guide, optimizer, 
              loss=Trace_ELBO())
    epochs = 40
    return svi, epochs

def train_model(trainloader, svi, epoch, device):
    '''
    forward + backward prop for 1 epoch
    prints the loss for every minibatch 
    '''
    running_loss = 0.0
    
    if dataset == "mnist" or dataset == "cifar10":
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # calculate the loss and take a gradient step
            running_loss += svi.step(inputs, labels)
            
            # print every 100 mini-batches, each minibatch has 64 images:
            if i % 100 == 99:    
                print('[epoch: %d, batch: %5d] loss/batch: %.3f' %
                    (epoch+1, i+1 , running_loss / 100))
                running_loss = 0.0
    
    if dataset in ["miniImageNet", "tieredImageNet", "CIFARFS", "FC100", 
                   "Omniglot"]:
        for i, batch in enumerate(trainloader):
            support_inputs, support_targets = batch["train"]
            query_inputs, query_targets = batch["test"]
            # episodic training:
            for support_input, support_target in zip(support_inputs, support_targets):
                support_input = support_input.to(device)
                support_target = support_target.to(device) 

                # calculate the loss and take a gradient step
                running_loss += svi.step(support_input, support_target)
            
            # print every 10 mini-batches:
            if i % 10 == 9:    
                print('[epoch: %d, batch: %5d] loss/batch: %.3f' %
                    (epoch+1, i+1 , running_loss / 10))
                running_loss = 0.0

def test_model(testloader, epoch, device):
    '''
    Tests the model accuracy over the test data in one epoch
    Prints the average loss
    '''
    def predict(x):
        '''
        samples a new neural network 10 times for making one prediction
        this gives us uncertainities on outputs
        to make a prediction, we average the  final layer output values 
        of the 10 sampled nets for the given input x
        '''
        num_forward_passes = 10
        sampled_models = [probabilistic_guide(None, None) 
                          for _ in range(num_forward_passes)]
        yhats = [model(x).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return np.argmax(mean.cpu().numpy(), axis=1)

    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = predict(images)
            total += labels.size(0)
            correct += (outputs == labels.cpu().numpy()).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%\n' % (
          100 * correct / total))

def train(dataset=dataset):
    '''
    Applies the train_model and test_model functions at each epoch
    '''
    # This loads the dataset and partitions it into batches:
    if dataset == "cifar10":
        trainset, testset = dp.load_cifar10()
        trainloader, testloader = dp.generate_batches(trainset, testset)
    if dataset == "mnist":
        trainset, testset = dp.load_mnist()
        trainloader, testloader = dp.generate_batches(trainset, testset)
    if dataset in ["miniImageNet", "tieredImageNet", "CIFARFS", "FC100", 
                   "Omniglot"]:
        meta_train = data_generator.generate_batch(test=False)
        meta_test = data_generator.generate_batch(test=True)
          
    # Loads the model and the training/testing functions:
    net = ResNet18(dataset)
    net, device = set_device(net)
    svi, epochs = inference()
    
    # Print the train and test accuracy after every epoch:
    if dataset == "mnist" or dataset == "cifar10":
        for epoch in range(epochs):
            train_model(trainloader, svi, epoch, device)
            test_model(testloader, epoch, device)
    
    if dataset in ["miniImageNet", "tieredImageNet", "CIFARFS", "FC100", 
                   "Omniglot"]:
        for epoch in range(epochs):
            train_model(meta_train, svi, epoch, device)
            test_model(meta_test, epoch, device) 
    
    print('Finished Training')   
    # Save the model:
    save_model(dataset, net)

def test(image_path):
    '''
    Classifies the image whose path entered on the terminal.
    '''
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 
               'frog', 'horse', 'ship', 'truck')
    img_tensor = dp.load_test_image(image_path).unsqueeze(0)
    net = ResNet18()
    load_model(net)
    # visualizes the outputs of the first CONV layer and saves in a file: 
    first_conv = net.conv1(img_tensor)
    first_conv = torchvision.utils.make_grid(first_conv, nrow=6, padding=8).\
                                                           detach().numpy()
    save_conv1(first_conv)
    # classifies the test image
    outputs = net(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted: %s" %classes[predicted[0]])

def save_conv1(img, N=6):
    '''
    Visualizes and saves the output of the first convolutional layer
    '''
    fig = plt.figure(figsize=(N, N))
    for i in range(img.shape[0]):
        ax1 = fig.add_subplot(N, N, i+1)
        ax1.imshow(img[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('CONV_rslt.png')    
    plt.show()


if __name__ == "__main__":
    fire.Fire()