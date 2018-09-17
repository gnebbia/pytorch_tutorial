# Pytorch Tutorial

Pytorch is a deep learning framework for python, although it's still at the moment (2018) more
used than tensorflow, it has a very interesting adoption curve amaong researchers.

In this tutorial we will go through the basics, and things which are not so clear in the common
tutorials found on the web (at least not so clear for me).

## Introduction

I would say that most pytorch programs contain a:

* A way to Load the data to feed to the neural network (from now on the **dataloader**)
* A Model of the Neural Network (from now on, the **model**)
* A Loss function to use in the backpropagation algorithm (from now on, the **loss**)
* An Optimizer for the Neural Network (from now on, the **optimizer**)
* A bunch of other neural network tuning variables (from now on, the **hyperparameters**)

Of course, these are only basic elements, then of course we have the training and validation phase 
or we can for example have regularization blocks or more advanced stuff.

In this tutorial, we will try to analyze each of these blocks, in order to be able to use pytorch
for most of the problems.


## Data Loading

When I started using pytorch, the surprising factor (for me) was the lack of tutorials on the web
which load a basic structured dataset (e.g., Iris dataset, or Titanic Kaggle dataset), this can be 
justified by the fact that the majority of the applications of deep learning is focused on unstructered
data, such as images, text, audio, and so on.

Anyway this does not mean, that we are not using deep learning on structured data, but still I had
many difficulties understanding how to load a basic structured dataset.

When I found out how data is actually loaded in pytorch, I thought that the approach was actually really
smart.

I will explain a very general approach, which is actually very flexible, and will allow us to load whatever
data we have to load.

In order to load data, we should use inheritance and write a data loading apposite class.

Let's say that we have our dataset which is loaded from a csv and we are using **pandas** (yeah I know, I
love pandas), we can write a class to load our dataset and in this class we have to specify:

* How we create our dataset, so the **init**ialization
* How we count elements in our dataset, so the **length**
* How to retrieve a sinngle element in our dataset, so a way to **get an item**

It's a very good exercise trying to write dataloaders for different datasets we meet.

Ok Let's see an example, let's start from a very basic dataset, which may be the IRIS dataset
from fisher, this dataset has 4 feature columns and 1 output which has three possible outputs,
specifically 'Setosa, Versicolor, Virginica'.

In order to write a dataloader for this basic dataset I would write:

```python
from torch.utils import data
import torch
import pandas as pd

class IrisDataset(data.Dataset):
  def __init__(self, df):
        # In this case, we are passing an entire dataframe 'df'
        self.df = df

  def __len__(self):
        # Denotes the total number of samples
        return len(self.df)

  def __getitem__(self, index):
        # Generates one sample of data
        # Select sample, in this case we want to be more general as possible
        # because 'index' could be an array, this particularly happens when we use
        # batches, then we have to give back both X and y, so features and labels
        X = np.array(self.df.iloc[index, 0:4].tolist(), dtype=np.float32)

        # If we are going to solve a classification problem we have to set the label
        # as an int variable or a long int variable.
        # For a regression problem we could have different types
        y = np.array(self.df.iloc[index, 4].tolist(), dtype=np.long)

        # Get data and get label
        return X, y
```

It can be a good idea to practice trying to write data loaders for different kinds of data,
in the case of structured data as we have seen, it is quite trivial, and we just have to point out which
is the label and which are the features.

I would personally put this dataloader in a file inside *project_name/main_package/dataloader.py*,
in a later example, we will put together all the pieces we are going to build, so for now,
let's just keep in mind that the data loader has been written.


## Model
The model in pytorch is what makes up the neural network architecture,
so here we choose the type and number of layers and also how many neurons each layer will have.

The types of layers we can use are:

* Fully Connected Layers
* Convolution Layers
* Recurrent Layers


Let's see some code to understand, I generally write my model in a separate file.

```python
import torch.nn as nn
import torch


class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, output_size):
        super(MyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layer1_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_layer2_size, hidden_layer2_size)
        self.relu3 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_layer2_size, output_size)

        # We do not add a Softmax layer, since this is automatically done,
        # if we use the CrossEntropyLoss function as a loss function
        # In cases where we do not use this specific loss function
        # we could simply add it as self.softmax = nn.LogSoftmax()
```

We can instantiate a model by executing:

```python
model = MyNetwork(4, 5, 5, 3)
```

We can also print our neural network architecture by executing:
```python
print(model)
```

## Loss Function

We can initialize a loss function with a single line of code, e.g.:

```python
# For a basic classification problem we would use:
criterion = nn.CrossEntropyLoss()
# This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

# In a basic regression problem instead we could use something like: 
criterion  = nn.MSELoss()
```

From time to time, it may happen that we have to write our own custom loss function,
in order to do this, I would suggest to write it in a different file,
in our project package to keep things clean, e.g., *project_name/main_package/myloss.py*.
and we just have to define the forward step, the backward will be automatically inferred and
computed.
We generally average the outpus before returning the loss array.



```python
import torch
import numpy as np

class DistanceLoss(torch.nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, dist_a, dist_b, tanh_output):
        x = tanh_output
        y = dist_b - dist_a

        # Normalize the difference between distances to be in the range [-1,1]
        y = (2*(y - dist_min)/(dist_max - dist_min)) -1
        y = y.view(-1,1)

        # To not have all the operations on a single line
        # I preferred to keep track of InterMediate steps
        # through variables called 'im'
        x_squared = torch.pow(x, 2)
        y_squared = torch.pow(y, 2)
        xy = x * y
        x2y = torch.mul(xy, 2)

        im1 = torch.add(x_squared, -y_squared)
        im2 = torch.add(im1, -x2y)

        im3 = torch.mul(im2, 50)
        
        loss = torch.add(im3, 100)

        loss = loss.mean()
```

## Optimizer
We will also need to define an optimizer, common choices without too much parameter
tuning is:

```python
lr = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#Another common option is:

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```
where model is the instantiated model we will see later, in an example where we will put
all the pieces together



## Putting Everything Together
Now let's put all the things together, let's say that e are building a neural
network classifier for the iris dataset, in this case, we will have,
our previously written dataloader class, with the neural network architecture
we wrote.

Now we just have to instantiate, the model (i.e., the neural network instance),
the loss function, set the batch size and shuffle the dataset.
```
# The current split is 95% of data is used for training and 5% for validation of the model
train = iris.sample(frac=0.70,random_state=200)
test = iris.drop(train.index)


# Here a custom Data Loader is used
train = IrisDataset(train)
test = IrisDataset(test)

# Generally we should choose the batch size as a multiple of 2
batch_size = 64
n_iters = 1000000

num_epochs = n_iters / (len(train) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)


train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

input_size = 4
hidden_size1 = 7
hidden_size2 = 8
num_classes = 3

model = MyNetwork(input_size, hidden_size1, hidden_size2, num_classes)
criterion = nn.CrossEntropyLoss()
```

Now we just have to train and validate our model.

### Training and Validation of the Model

```python
for epoch in range(num_epochs):
    for i, (X, labels) in enumerate(train_loader):
        X = Variable(X)
        
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = model(X)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iter += 1
        
        # we want to check the accuracy with test dataset every 500 iterations
        # we can change this number, it is just if it is too small we lose a lot of time
        # checking accuracy while if it is big, we have less answers but takes less time for the algorithm
        if iter % 100 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            
            # iterate through test dataset
            for X, labels in test_loader:
                X = Variable(images)
                
                outputs = model(images)
                # get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            
            print("Iteration: {}. Loss: {}. Accuracy: {}".format(iter, loss.item(), accuracy))
```



## Appendix A: Writing Data Loaders for fun and practice

```python
import torch
import pandas as pd
from torch.utils import data

class Dataset(data.Dataset):
  def __init__(self, df):
        # but we are interested only in certainn columns (i.e., fields),
        # this is a scenario for example where some of the columns in our initial dataset
        # are metadata or information which we do not need/want to use in our training/validation
        user_1_features = [0,1,2,3,4,5,6,7,8,9,10,11,15]
        user_2_features = [0,1,2,3,4,5,16,17,18,19,20,21,22]
        self.user_1 = df.iloc[:, user_1_features]
        self.user_2 = df.iloc[:, user_2_features]
        self.df = df

  def __len__(self):
        # Denotes the total number of samples
        return len(self.df)

  def __getitem__(self, index):
        # Generates one sample of data
        # Select sample, in this case we want to be more general as possible
        # because 'index' could be an array, this particularly happens when we use
        # batches
        user_1_data = np.array(self.user_1.iloc[index].tolist(), dtype=np.float32)
        user_2_data = np.array(self.user_2.iloc[index].tolist(), dtype=np.float32)
        user_1_dist = np.array(self.user_1.iloc[index]["distance_1"].tolist(), dtype=np.float32)
        user_2_dist = np.array(self.user_2.iloc[index]["distance_2"].tolist(), dtype=np.float32)

        # Load data and get label, this is what we are returning back to the user
        return user_1_data, user_2_data, user_1_dist, user_2_dist
```
