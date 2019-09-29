
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

