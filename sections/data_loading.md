
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


