import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import torchvision.models as models
import numpy as np
from sklearn.metrics import classification_report
from torchvision.models import VisionTransformer
import datetime;

import timm
from timm.models.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm 
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ct stores current time
ct = datetime.datetime.now()
print("current time:-", ct)


import time



# Define the dataset and transformations
transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



transformdeit = transforms.Compose([
    transforms.RandomCrop(224, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])




train_dataset = torchvision.datasets.ImageFolder(root='/home/user/dataset-inaturalist/mattonelle256/train', transform=transformdeit)
val_dataset = torchvision.datasets.ImageFolder(root='/home/user/dataset-inaturalist/mattonelle256/val', transform=transformdeit)

test_dataset = torchvision.datasets.ImageFolder(root='/home/user/dataset-inaturalist/mattonelle256/testmattonelle-baseline1x', transform=transformdeit)

# Define the dataloaders  
train_loader = DataLoader(train_dataset, batch_size=950, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=660, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=120, shuffle=True, num_workers=2)

device="cuda:0"


# LOAD MODEL FROM TORCH HUB OR FROM A PREVIOUS TRAINING

model = models.vit_b_16(pretrained=True)

#model = torch.load('/tmp/last-VIT-BASE.pth')



from vit_pytorch import ViT


# start from 
epocarestart=2


# VIT - freeze some layer 


for param in model.parameters():
        param.requires_grad = False






# add dropout 

for name, layer in model.named_modules():
    if isinstance(layer, nn.Dropout):
         layer.p = 0.2

#some custom heads 

heads = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=1024, bias=True),
         nn.LeakyReLU(),
        nn.Dropout(0.1),
        torch.nn.Linear(in_features=1024, out_features=9, bias=True),

    )

heads2024 = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=2048, bias=True),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        torch.nn.Linear(in_features=2048, out_features=9, bias=True),

    )



heads2 = nn.Sequential(
    nn.Linear(1024, 512),
    nn.LeakyReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(512, 9)
)


heads3 = nn.Sequential(
    nn.Linear(768, 1024),
    nn.LeakyReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(1024, 9)
)


heads4 = nn.Sequential(
    nn.Linear(192, 1024),
    nn.LeakyReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(1024, 9)
)



heads5 = nn.Sequential(
    nn.Linear(384, 1024),
    nn.LeakyReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(1024, 9)
)

print ("MODIIED MODEL")
print(model)


model.heads = heads2024



# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()


model = model.to(device)



optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.007)


# LR scheduled

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)



# lr def e-4
print(model)



# visualizza numero parametri
print("parameters ")
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'parameters num: {num_params}')




# Train the model 
num_epochs = 50
#num_epochs = 10
train_losses, val_losses, train_acc= [], [], []
val_acc =[]
val_losses2= []
print("start train")
for epoch in range(epocarestart,num_epochs):
    print("epoch " + str(epoch))
    running_loss = 0.0
    running_corrects=0.0
    for i, data in enumerate(train_loader):
        if (i % 79 ==0):
          print("batch" + str(i))
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        model.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

      #  loss.requires_grad = True
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    # Print training loss
    print(f'Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}')
    train_losses.append(running_loss / len(train_loader))
#    train_losses.append(running_loss / len(train_loader).to_list())
    epoch_acc = running_corrects.double() / len(train_loader)
    tmpepoch_acc=epoch_acc.item()
    print(type(train_acc))
    train_acc.append(tmpepoch_acc)
    print(train_acc)
#    train_acc.append(epoch_acc.tolist())
    print("train acc=",epoch_acc)
    #train_acc.append(epoca_acc.tolist())
    print("start validation" )
    # Evaluate the model on the validation set
    model.eval()
    preds1 = []
    targets1 = []
    val_loss=0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            outputs = model(inputs)
            if ((outputs.argmax(dim=1).cpu().numpy())[0] < 10):
             preds1.append(outputs.argmax(dim=1).cpu().numpy())
             targets1.append(labels.cpu().numpy())
             loss2 =criterion(outputs, labels)
             val_loss += loss2.item()
    preds1 = np.concatenate(preds1)
    targets1 = np.concatenate(targets1)
    scheduler.step()

    # Calculate accuracy and f1-score
    accuracy = accuracy_score(targets1, preds1)
    f1 = f1_score(targets1, preds1, average='weighted')
    print("num elementi val loader = " , len(val_loader))
    print('Validation Loss: {:.4f}'.format(val_loss / len(val_loader)))

    print(classification_report(targets1, preds1))
    print(f'Epoch {epoch+1} - Validation accuracy: {accuracy}')
    print(f'Epoch {epoch+1} - Validation f1-score: {f1}')
    print("VAL ACC TYPE =",type(val_acc))
    val_acc.append(accuracy/len(val_loader))
    print("VAL ACC TYPE  after append=",type(val_acc))

    valoreloss=(val_loss / len(val_loader))
    print("valore losses= " ,valoreloss)

    val_losses2.append(valoreloss)
    print("val losses= ", val_losses2)
    print("val ACC= ", val_acc)

   # ct stores current time
    ct = datetime.datetime.now()
    print("current time:-", ct)

    torch.save(model, "last-VIT-BASE-Tile.pth")



    print("Start TEST with new data")

    # Evaluate the model on the test set
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            outputs = model(inputs)
            if ((outputs.argmax(dim=1).cpu().numpy())[0] < 10):
              preds.append(outputs.argmax(dim=1).cpu().numpy())
              targets.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    print("TEST done!")
    # Calculate accuracy and f1-score
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    classes= (0,1,2,3,4,5,6,7,8)
    # Build confusion matrix
    cf_matrix = confusion_matrix(targets, preds)
    df_cm = pd.DataFrame(cf_matrix).astype(int)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    nomematricefile="confusionmap-VIT-Base-Tile.png"
    plt.savefig(nomematricefile)




    print(classification_report(targets, preds))

    # Print accuracy and f1-score for test set
    print(f'Epoch {epoch+1} - Test accuracy: {accuracy}')
    print(f'Epoch {epoch+1} - Test f1-score: {f1}')
    ct = datetime.datetime.now()
    print("current time:-", ct)

    # Save the model checkpoint
    #if (epoch % 3==0):
    torch.save(model, "UAV4Tree2-Last-VIT-Tile.pth")

print(" EPOCHS number =" + str(num_epochs))
if (1==1):
    # plot loss
    stringapergrafici="Grafico-VIT-"
    plt.figure(figsize=(10, 5))
    print("TRAIN LOSSES",train_losses)
    print(val_losses2)
    plt.title("Training and Validation Loss")
    plt.plot(val_losses2, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    nomefileloss = stringapergrafici +"-LOSS.png"
    plt.savefig(nomefileloss)
    #plt.show()



    #plot acc
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    print ("vettore acc")
    train_acc=torch.tensor(train_acc).detach().cpu().numpy()

    val_acc=torch.tensor(val_acc).detach().cpu().numpy()

    plt.plot(train_acc,label="train")
    plt.plot(val_acc, label="val")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    nomefileacc=stringapergrafici+"-ACC.png"
    plt.savefig('acc.png')
    plt.savefig(nomefileacc)
    plt.show()

