
Pytorch is a deep learning framework for python, although it's still at the moment (2019) more
used than tensorflow, it has a very interesting adoption curve amaong researchers.

In this tutorial we will go through the basics, and things which are not so clear in the common
tutorials found on the web (at least not so clear for me).

Let's see in these notes how pytorch works.

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


