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



