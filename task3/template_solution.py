# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
from math import floor
import random
import numpy as np
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

embedding_size_global = 2048
input_scaler = 0
output_scaler = 0
torch.manual_seed(3473)

# not tested yet. should be able to run geneeate embeddings (if there are not bugs, which there probably are)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    print("-- genearte embeddings --")

    # turns an image into a PyTorch tensor (matrix where each entry is a vector (e.g., for rgb information))
    # according to PyTorch (see https://pytorch.org/vision/stable/models.html), "All the necessary information 
    # for the inference transforms of each pre-trained model is provided on its weights documentation"
    # This is because only if the input data matches with the format of the data that was used for training
    # do we have a certain quality guarantee of the output (here the embeddings) 
    # need to check at https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    # how to correctly preprocess, right now, we are just using transforms.ToTensor
    train_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    train_dataset_imgs = datasets.ImageFolder(root="dataset/")

    #train_dataset = datasets.ImageFolder(root="dataset/")

    print("-- loaded data set --")

    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=16)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    # model = nn.Module()
    embeddings = []
    # embedding_size = 1000 # Dummy variable, replace with the actual embedding size once you 

   # Useing pretrained model ResNet for now

    print("-- modifiying pretrained model --")

    # here we define the pre-trained model obtained from PyTorch
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    newmodel.eval() #prepares the new model for evaluation (not all models need this)


    embedding_size = embedding_size_global # Dummy variable, replace with the actual embedding size once you 
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    preprocess = weights.transforms()

    print("-- preprocessing --")

    print(train_loader)

    #this is the slow way of doing it
    for i,(a,b) in enumerate(train_dataset): 
        if(i % 100 == 0): print(i)
        batch = preprocess(a).unsqueeze(0)
        result = newmodel(batch).detach()
        # transform each
        embeddings[i] = result.numpy().reshape((embedding_size)) 
   

    # for [a,b] in train_loader:
    #     print(a,b)

    # for i, x in enumerate(train_loader):
    #     if(i % 1000): print(i)

    #     batch = preprocess(a).unsqueeze(0)
    #     result = newmodel(batch).detach()
    #     embeddings[i*train_loader.batch_size+j] = result.numpy().reshape((embedding_size)) 

    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')
    # TODO: Normalize the embeddings across the dataset
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)

    global input_scaler
    input_scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = input_scaler.transform(X)
    return X_scaled, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # self.fc = nn.Linear(3000, 1)

        self.hidden = nn.Linear(embedding_size_global*3, 20)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(20)
        self.hidden1 = nn.Linear(20, 20)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(20)
        self.hidden2 = nn.Linear(20, 20)
        self.relu2 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(20)
        self.out = nn.Linear(20, 1) 
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(self.bn1(x)))
        x = F.relu(self.hidden2(self.bn2(x)))
        x = F.sigmoid(self.out(self.bn3(x)))
        return x

def train_model(train_loader, validate):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs =5 
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.

    loss_function = nn.L1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_len = len(train_loader) 
    # indices = range(data_len)
    train_portion = 0.8 if validate else 1 
    # train_idx = random.sample(indices, k=floor(train_portion*data_len))
    # test_idx = list(set(indices) - set(train_idx)) 
    # train_split = [(data, target) for idx, (data, target) in enumerate(train_loader) if idx in train_idx]
    # val_split = [(data, target) for idx, (data, target) in enumerate(train_loader) if idx in test_idx]
    train_split = [(data, target) for idx, (data, target) in enumerate(train_loader) if idx <= floor(data_len*train_portion)]
    val_split = [(data, target) for idx, (data, target) in enumerate(train_loader) if idx > floor(data_len*train_portion)]
    predictions = [0]*len(val_split)
    for epoch in range(n_epochs):        
        for data, target in train_split:
            optimizer.zero_grad()
            output = model(data)
            # loss = loss_function(torch.squeeze(output), target.to(torch.float32))
            loss = loss_function(torch.squeeze(output), target)
            loss.backward()
            optimizer.step()

        if validate: 
            model.eval()
            for i, (data, target) in enumerate(val_split): 
                val_out = model(data)
                predictions[i] = get_correct_predictions(torch.squeeze(val_out), target)

            correct_total = [sum(x) for x in zip(*predictions)]
            print('Epoch {}, accuracy {}'.format(epoch, correct_total[0]/correct_total[1]))

    print("-- finished training --")
    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: the predictions
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)

    return predictions

def get_accuracy(predictions, target):
    '''
    IN: output : torch.Tensor([n])
        target : torch.Tensor([n])
    OUT: (#correct_predictions, n)
    '''
    output = predictions.clone().detach()
    torch.logical_xor(target,predictions,out=output)
    torch.logical_not(output,out=output)
    return torch.sum(output)/output.size(dim=0)

def evaluate_model(model, loader, y):
    predictions = test_model(model, loader)
    predictions = torch.from_numpy(predictions)

    accuracy = get_accuracy(predictions.squeeze(), y)
    # l1_loss = np.sum(abs(predictions.squeeze() - y))/len(y)

    # print("Have a loss of",l1_loss)
    print("Accuracy on validation set",accuracy)


def test_and_save(model, loader):
    """
    runs test model and saves the predictions
    """

    predictions = test_model(model, loader)
    np.savetxt("results.txt", predictions, fmt='%i')



# Main function. You don't have to change this
if __name__ == '__main__':
    print("-- main function running --")

    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    print("-- loaded the data --")


    doLocalTesting = False 
    if(doLocalTesting): # Do testing
        p = 0.8
        length = y.shape[0]
        train_loader = create_loader_from_np(X[:int(length*p)], y[:int(length*p)], train = True, batch_size=64)
        test_loader = create_loader_from_np(X[int(length*p):], train = False, batch_size=2048, shuffle=False)

        model = train_model(train_loader)

        print("-- trained model --")

        evaluate_model(model, test_loader, torch.from_numpy(y[int(length*p):]))

    else:
        # Create data loaders for the training and testing data
        train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
        test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)


        # define a model and train it
        _ = train_model(train_loader, validate=True)
        # model = train_model(train_loader, validate=False)

        print("-- trained model --")
        
        # test the model on the test data
        # test_and_save(model, test_loader)
        print("Results saved to results.txt")
