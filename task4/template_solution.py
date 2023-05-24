# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the todo gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error

np.random.seed(11239)
torch.manual_seed(11239)


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
        # todo: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        embedding_size = 30

        self.fc1 = nn.Linear(in_features,50)
        self.fc2a = nn.Linear(50, 50)
        self.fc2b = nn.Linear(50, embedding_size)

        self.dropout = nn.Dropout(0.5)
        #self.fc4 = nn.Linear(embedding_size, 50)

        self.out = nn.Linear(embedding_size, 1)



    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        # todo: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.

        x = self.get_embeddings(x)
        #x = self.dropout(x)
        #x = F.relu(self.fc4(x))
        x = self.out(x)

        return x
    
    def get_embeddings(self, x):
        
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2a(x))
        x = F.sigmoid(self.fc2b(x))

        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
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
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # assuming x_tr: x_train and x_val: x_evalulate

    # model declaration
    model = Net(in_features)
    model.train()
    
    # todo: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    # === Training the Model ===
    criterion = nn.L1Loss()
    valildationLoss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    losses = []

    for i in range(epochs):
        y_pred = model.forward(x_tr).squeeze()
        loss = criterion(y_pred, y_tr)
        losses.append(loss)

        if(i % 10 == 0):
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_test = model.forward(x_val).squeeze()
        # if(i % 10 == 0):
        #     print(f"\t\tWe have a loss of {valildationLoss(y_pred_test, y_val):10.8f}")


    print(f"\t\tWe have a loss of {valildationLoss(y_pred_test, y_val):10.8f}")

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # todo: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
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
    # todo: Implement the regression model. It should be able to be trained on the features extracted
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

    x_embeddings = np.apply_along_axis(temp, 1, x_train)


    #model = HuberRegressor(alpha=1, max_iter = 1000, epsilon=1) #TODO, this could be better
    split_K = 80


    # =========================================================================================================
    # attempt with gaussian kernal, hasn't worked yet
    # print("========= White Kernel =========")

    # for noise_level_local in [5,3,2,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005]:
    #     print("Noise Level:",noise_level_local)
    #     #kernel = 1.0 * RBF(1.0)
    #     #kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    #     kernel = DotProduct() + WhiteKernel(noise_level=noise_level_local)

    #     gpc = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_embeddings[0:split_K], y_train[0:split_K])


    #     #model.fit(x_embeddings[0:10], y_train[0:10])

    #     print("  score trained:",1-gpc.score(x_embeddings[0:split_K], y_train[0:split_K]))
    #     print("  score unseen: ",1-gpc.score(x_embeddings[split_K:], y_train[split_K:]))
            

    #     # train it on the whole set
    #     gpc2 = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_embeddings, y_train)

    #     print("  score mode2:  ",1-gpc2.score(x_embeddings, y_train))

    #     y_pred = gpc2.predict(np.apply_along_axis(temp, 1, x_test))



    # =========================================================================================================
    print("========= Linear Regression =========")

    model = LinearRegression().fit(x_embeddings[0:split_K], y_train[0:split_K])

    #model.fit(x_embeddings[0:10], y_train[0:10])

    print("score trained:",1-model.score(x_embeddings[:split_K], y_train[:split_K]))
    print("score unseen: ",1-model.score(x_embeddings[split_K:], y_train[split_K:]))
    y_train_pred = model.predict(x_embeddings[split_K:])
    print("score rmse:   ",mean_squared_error(y_train_pred,y_train[split_K:]))


    # train it on the whole set
    model2 = LinearRegression().fit(x_embeddings, y_train)
    print("score mode2:  ",1-model2.score(x_embeddings, y_train))
    y_pred = model2.predict(np.apply_along_axis(temp, 1, x_test))

    # scores = cross_val_score(model, x_embeddings, y_train, scoring="neg_mean_squared_error", cv=10)
    # print(scores)

    # =========================================================================================================
    print("========= Ridge Regression =========")
    alpha = 0.5

    model = Ridge(alpha=alpha).fit(x_embeddings[0:split_K], y_train[0:split_K])

    #model.fit(x_embeddings[0:10], y_train[0:10])

    print("score trained:",1-model.score(x_embeddings[:split_K], y_train[:split_K]))
    print("score unseen: ",1-model.score(x_embeddings[split_K:], y_train[split_K:]))
          
    y_train_pred = model.predict(x_embeddings[split_K:])
    print("score rmse:   ",mean_squared_error(y_train_pred,y_train[split_K:]))


    # train it on the whole set
    model2 = Ridge(alpha=alpha).fit(x_embeddings, y_train)
    print("score mode2:  ",1-model2.score(x_embeddings, y_train))



    y_pred = model2.predict(np.apply_along_axis(temp, 1, x_test))

    # scores = cross_val_score(model, x_embeddings, y_train, scoring="neg_mean_squared_error", cv=10)
    # print(scores)

    print("-- saving the data --")

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")