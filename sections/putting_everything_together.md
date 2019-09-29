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

## Training and Validation of the Model

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



