# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
from math import ceil
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

np.random.seed(11234)
torch.manual_seed(11234)


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, in_features):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        self.in_features = in_features
        embedding_size = 20  
        self.fc1 = nn.Linear(in_features, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, embedding_size)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(embedding_size, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.get_embeddings(x)
        x = self.out(x)

        return x
    
    def get_embeddings(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=10000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    print("-- in make_feature_extractor --")

    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    # print("x_pretrain:")
    # print(x_tr[0,:])

    # Standarize data 
    global input_scaler
    input_scaler = StandardScaler()
    input_scaler.fit(x_tr)
    x_tr = input_scaler.transform(x_tr)
    x_val = input_scaler.transform(x_val)
    # print("x_pretrain after standardizing:")
    # print(x_tr[0,:])


    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # create loader for training data (to escape local minima during training)
    train_dataset = TensorDataset(x_tr.type(torch.float), y_tr.type(torch.float))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # model declaration
    model = Net(in_features)
    model.train()
    
    # todo: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    # === Training the Model ===
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10 
    tr_losses = [0.]*epochs
    val_losses = [0.]*epochs

    # train model
    loss_per_batch = (ceil(x_tr.shape[0]/batch_size)) * [0]
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(torch.squeeze(output), target.to(torch.float32))
            # loss = criterion(output, target.to(torch.float32))
            loss.backward()
            optimizer.step()
            loss_per_batch[batch_idx] = loss.item()

        model.eval()
        y_pred_test = model.forward(x_val).squeeze()
        val_loss = criterion(y_pred_test, y_val).item()

        training_loss = sum(loss_per_batch)/len(loss_per_batch)
        tr_losses[epoch] = training_loss
        val_losses[epoch] = val_loss
        print(f"training loss   {training_loss:10.8f}")
        print(f"validation loss {val_loss:10.8f} \n")

    # plot loss over training epochs
    plt.plot(range(epochs), tr_losses, label='Training loss')
    plt.plot(range(epochs), val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and validation loss over time')
    plt.legend()
    plt.show()

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        return model.get_embeddings(x)

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = HuberRegressor(alpha=1, max_iter = 1000, epsilon=1) #TODO, this could be better
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    print("-- starting in main --")
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("-- data loaded! --")

    # Shapes: (05k, 1k) (50k) train: (100, 1k) (100) (10k, 1k)


    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)

    print("-- features extractor done --")

    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])

    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    print("-- fitting part II --")


    def temp(x):
        x = torch.tensor(x, dtype=torch.float)
        x = feature_extractor(x)
        return x.detach().numpy()

    # extract embeddings for train- and test-data
    x_test_orig = x_test
    x_train_orig = x_train
    x_train = np.apply_along_axis(temp, 1, input_scaler.transform(x_train))
    x_test = np.apply_along_axis(temp, 1, input_scaler.transform(x_test))

    # normalize embedded data (again, recommended for GPR)
    GPR_input_scaler = preprocessing.StandardScaler().fit(x_train)
    old_y_train_shape = y_train.shape
    x_train = GPR_input_scaler.transform(x_train)
    x_test = GPR_input_scaler.transform(x_test)
    rows, _ = x_train.shape
    y_train_reshaped = y_train.reshape((rows, 1))
    output_scaler = preprocessing.StandardScaler().fit(y_train_reshaped) 
    y_train = output_scaler.transform(y_train_reshaped).reshape(old_y_train_shape)

    # ls_list = [1e-4, 1e-3, 1e-2, 1e-1]
    noise_list = [1e-4, 1e-3, 1e-2, 1e-1]
    # ls = np.full((10,), 1e-1) # best length_scale found empirically (only one that converges)
    ls = 1e-1
    k = 5 
    mean_scores = list(noise_list)

    for i, ns in enumerate(noise_list):
        print("Using noise = {}".format(ns))
        kernel = RBF(length_scale=ls) + WhiteKernel(ns)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0) 
        scores = cross_val_score(gpr, x_train, y_train, cv=k)
        mean_scores[i] = scores.mean()
        print("Got mean score of {} for noise = {}".format(mean_scores[i], ns))

    best_ns = noise_list[mean_scores.index(max(mean_scores))]
    print("best noise = {}".format(best_ns))

    gpr = GaussianProcessRegressor(kernel=RBF(length_scale=ls) + WhiteKernel(best_ns), random_state=0).fit(x_train, y_train)
    y_pred_scaled = gpr.predict(x_test)
    # need to rescale since the Gaussian regressor was trained on standardized output (zero mean, unit variance)
    rows, _ = x_test.shape
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape((rows,1))) 
    y_pred = y_pred.reshape(y_pred_scaled.shape)

    print("-- saving the data --")

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test_orig.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")