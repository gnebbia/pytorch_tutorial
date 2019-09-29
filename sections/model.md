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

