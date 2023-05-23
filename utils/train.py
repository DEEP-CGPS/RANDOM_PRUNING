import os
import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from utils.custom_dataset import dataset_list, CustomDataset


##===================================================================================##
##===================================================================================##
##===================================================================================##


def get_model(num_classes, args):
    if args.model_architecture == "ResNet18":
        model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
        num_fltrs = model.fc.in_features
        model.fc = nn.Linear(num_fltrs, num_classes)
    elif args.model_architecture == "VGG16":
        model = torchvision.models.vgg16_bn(weights="VGG16_BN_Weights.IMAGENET1K_V1") 
        model.classifier[6] = nn.Linear(4096,num_classes)
    return model


##===================================================================================##
##===================================================================================##
##===================================================================================##


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
        num_classes = len(trainset.classes)
        
        
    elif args.dataset == "Tomato_Leaves":
        data_dir = "./data/Tomato_Leaves"
        train_list, test_list, class_names = dataset_list(data_dir)
        num_classes = len(class_names)
        
        trainset = CustomDataset(train_list,transform)
        testset = CustomDataset(test_list,transform)  
        
        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
        
    return train_loader,test_loader, num_classes
 
        


##===================================================================================##
##===================================================================================##
##===================================================================================##


def train_epoch(model, device, data_loader, criterion, optimizer, eval_metric, num_classes = 0):
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
    total_labels = []
    total_outs = []
    model.train()
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        output = model(inputs)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        loss = criterion(output, labels)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_labels.append(labels)
        total_outs.append(predicted)
        
    
    train_loss = running_loss / len(data_loader)
    
    if eval_metric == "accuracy":
        train_acc = 100 * train_correct / total
        return train_loss, train_acc
    
    elif eval_metric == "f1_score":
        total_labels = torch.cat(total_labels, dim=0)
        total_outs = torch.cat(total_outs, dim=0)
        total_outs[total_outs > num_classes-1] = num_classes-1
        
        train_f1s = multiclass_f1_score(total_outs, total_labels, num_classes = num_classes, average = "macro")
        #accuracy = multiclass_accuracy(total_outs, total_labels, num_classes = num_classes) * 100
        return train_loss, train_f1s
    

##===================================================================================##
##===================================================================================##
##===================================================================================##


def test_epoch(model, device, data_loader, criterion, eval_metric, num_classes = 0):
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
    total_labels = []
    total_outs = []
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        output = model(inputs)
        _, predicted = torch.max(output.data,1)
        total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
        loss = criterion(output,labels)
        running_loss += loss.item()
        
        total_labels.append(labels)
        total_outs.append(predicted)
    
    
    val_loss = running_loss / len(data_loader)
        
    if eval_metric == "accuracy":
        val_acc = 100 * val_correct / total
        return val_loss, val_acc
    
    elif eval_metric == "f1_score":
        total_labels = torch.cat(total_labels, dim=0)
        total_outs = torch.cat(total_outs, dim=0)
        total_outs[total_outs > num_classes-1] = num_classes-1
        
        val_f1s = multiclass_f1_score(total_outs.cpu(), total_labels.cpu(), num_classes = num_classes, average = "macro")
        #accuracy = multiclass_accuracy(total_outs, total_labels, num_classes = num_classes) * 100
        return val_loss, val_f1s
    

##===================================================================================##
##===================================================================================##
##===================================================================================##


def train_model(args,
                train_loader = None,
                test_loader = None,
                model = None,
                num_classes = 0):
    
    if not os.path.exists("models"):
        os.makedirs("models")

    model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    if args.optimizer_val == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)
    elif args.optimizer_val == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)

    # Training Loop
    best_model_acc = 0
    best_model_f1s = 0
    
    start_time = time.time()
    for epoch in range(args.num_epochs):
        
        
        if args.eval_metric == "accuracy":
            
            train_loss, train_acc = train_epoch(model, args.device, train_loader, criterion, optimizer, args.eval_metric)
            test_loss, test_acc = test_epoch(model, args.device, test_loader, criterion, args.eval_metric)

            end_time = time.time() - start_time
            
            print(f"Epoch: [{epoch + 1}/{args.num_epochs}]\t || Training Loss: {train_loss:.3f}\t || Val Loss: {test_loss:.3f}\t || Training Acc: {train_acc:.2f}% \t ||  Val Acc: {test_acc:.2f}% \t || Time: {time.strftime('%H:%M:%S', time.gmtime(end_time))}")
            
            if best_model_acc < test_acc:
                best_model_acc = test_acc
                model_name = f'{args.model_architecture}_{args.dataset}_{args.model_type}'
                print(f"Model Name: {model_name}")
                torch.save(model,f'models/{model_name}.pth')
            
            
        
        elif args.eval_metric == "f1_score":
            
            train_loss, train_f1s = train_epoch(model, args.device, train_loader, criterion, optimizer, args.eval_metric, num_classes = num_classes)
            test_loss, test_f1s = test_epoch(model, args.device, test_loader, criterion, args.eval_metric, num_classes = num_classes)

            end_time = time.time() - start_time
            
            print(f"Epoch: [{epoch + 1}/{args.num_epochs}]\t || Training Loss: {train_loss:.3f}\t || Val Loss: {test_loss:.3f}\t || Training F1-score: {train_f1s:.3f} \t ||  Val F1-score: {test_f1s:.3f} \t || Time: {time.strftime('%H:%M:%S', time.gmtime(end_time))}")
        
            if best_model_f1s < test_f1s:
                best_model_f1s = test_f1s
                model_name = f'{args.model_architecture}_{args.dataset}_{args.model_type}'
                print(f"Model Name: {model_name}")
                torch.save(model,f'models/{model_name}.pth')
        
        
