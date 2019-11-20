import torch 

def save_model(dataset, net):
    '''
    Saves the model to the directory Model
    '''
    if dataset == "cifar10":
        torch.save(net.state_dict(), f="Model/cifar10_model.model")
    if dataset == "mnist":
        torch.save(net.state_dict(), f="Model/mnist_model.model")
    if dataset == "Omniglot":
        torch.save(net.state_dict(), f="Model/omniglot_model.model")
    if dataset == "miniImageNet":
        torch.save(net.state_dict(), f="Model/miniImageNet_model.model")
    if dataset == "tieredImageNet":
        torch.save(net.state_dict(), f="Model/tieredImageNet_model.model")
    if dataset == "FC100":
        torch.save(net.state_dict(), f="Model/FC100_model.model")
    if dataset == "CIFARFS":
        torch.save(net.state_dict(), f="Model/CIFARFS_model.model")
    print("Model saved successfully.")

def load_model(dataset, net):
    '''
    Loads the network trained by GPU to CPU for inference. 
    '''
    try:
        if dataset == "cifar10":
            net.load_state_dict(torch.load("Model/cifar10_model.model", 
                                            map_location='cpu'))
        if dataset == "mnist":
            net.load_state_dict(torch.load("Model/mnist_model.model", 
                                            map_location='cpu'))
        if dataset == "Omniglot":
            net.load_state_dict(torch.load("Model/omniglot_model.model", 
                                            map_location='cpu'))
        if dataset == "miniImageNet":
            net.load_state_dict(torch.load("Model/miniImageNet_model.model", 
                                            map_location='cpu'))
        if dataset == "tieredImageNet":
            net.load_state_dict(torch.load("Model/tieredImageNet_model.model", 
                                            map_location='cpu'))
        if dataset == "FC100":
            net.load_state_dict(torch.load("Model/FC100_model.model", 
                                            map_location='cpu'))
        if dataset == "CIFARFS":
            net.load_state_dict(torch.load("Model/CIFARFS_model.model", 
                                            map_location='cpu'))                                            
    except RuntimeError:
        print("Runtime Error!")
        print(("Saved model must have the same network architecture with"
               " the CopyModel.\nRe-train and save again or fix the" 
               " architecture of CopyModel."))
        exit(1) # stop execution with error

def set_device(net):
    '''
    Trains network using GPU, if available. Otherwise uses CPU.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on: %s\n" %device)
    # .double() will make sure that MLP will process tensor
    # of type torch.DoubleTensor:
    return net.to(device), device