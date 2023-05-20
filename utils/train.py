import os
import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score, binary_accuracy
from utils.custom_dataset import dataset_list, CustomDataset

def get_model(args):
    if args.model_architecture == "ResNet18":
        model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
    elif args.model_architecture == "VGG16":
        model = torchvision.models.vgg16_bn(weights="VGG16_BN_Weights.IMAGENET1K_V1") 
    return model


def get_dataset(args):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if args.dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    elif args.dataset == "Tomato_Leaves":
        data_dir = "./data/Tomato_Leaves"
        train_list, test_list, class_names = dataset_list(data_dir)
        
        
        trainset = CustomDataset(train_list,transform)
        testset = CustomDataset(test_list,transform)       
 
        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
    return train_loader,test_loader


def train_epoch(model, device, data_loader, criterion, optimizer):
    """train_loss, accuracy = train_epoch(model, device, data_loader, criterion, optimizer)
    
    Function for each training epoch.
    
    Parameters:
        model = pytorch network model
        device = torch.device() object to use GPU or CPU during the training
        data_loader = Pytorch DataLoader object
        criterion = Pytorch loss function applied to the model
        optimizer = Pytorch optimizer applied to the model
    
    Returns:
        train_loss = float average training loss for one epoch
        accuracy = float average training accuracy for one epoch
    """
    
    train_correct = 0
    running_loss = 0.0
    total = 0
    model.train()
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    accuracy = 100 * train_correct / total
    train_loss = running_loss / len(data_loader)
    
    return train_loss, accuracy




def test_epoch(model, device, data_loader, criterion):
    """train_loss, accuracy = validation_epoch(model, device, data_loader, criterion)
    
    Function for each training epoch.
    
    Parameters:
        model = pytorch network model
        device = torch.device() object to use GPU or CPU during the training
        data_loader = Pytorch DataLoader object
        criterion = Pytorch loss function applied to the model
    
    Returns:
        val_loss = float validation loss for one epoch
        val_correct = integer number of correct predictions for one epoch
    """
    val_loss, val_correct = 0.0, 0
    model.eval()
    running_loss = 0
    total = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        output = model(inputs)
        loss = criterion(output,labels)
        total += labels.size(0)
        running_loss += loss.item()
        _, predictions = torch.max(output.data,1)
        val_correct += (predictions == labels).sum().item()
        
    val_acc = 100 * val_correct / total
    val_loss = running_loss / len(data_loader)
    
    return val_loss, val_acc

def train_model(args,
                train_loader = None,
                test_loader = None,
                model = None
                 ):
    
    if not os.path.exists("models"):
        os.makedirs("models")

    model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    if args.optimizer_val == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)
    elif args.optimizer_val == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)

    # Training Loop
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    best_model_acc = 0
    
    start_time = time.time()
    for epoch in range(args.num_epochs):
        
        train_loss, train_acc = train_epoch(model, args.device, train_loader, criterion, optimizer)
        test_loss, test_acc = test_epoch(model,  args.device, test_loader, criterion)

        end_time = time.time() - start_time
        
        print(f"Epoch: [{epoch + 1}/{args.num_epochs}]\t || Training Loss: {train_loss:.3f}\t || Val Loss: {test_loss:.3f}\t || Training Acc: {train_acc:.2f}% \t ||  Val Acc: {test_acc:.2f}% \t || Time: {time.strftime('%H:%M:%S', time.gmtime(end_time))}")

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc) 

        if best_model_acc < test_acc:
            best_model_acc = test_acc
            model_name = f'{args.model_architecture}_{args.dataset}_{args.model_type}'
            print(f"Model Name: {model_name}")
            torch.save(model,f'models/{model_name}.pth')
