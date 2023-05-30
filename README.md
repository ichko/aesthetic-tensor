# 👨‍🎨 Aesthetic-Tensor

A simple fluent API for viewing tensors.

Instead of trying to decipher this

```py
T = torch.rand(3, 100, 100)
T
```

```py
>> tensor([[[0.9531, 0.3449, 0.8426,  ..., 0.3743, 0.4693, 0.6880],
            [0.2732, 0.3456, 0.6288,  ..., 0.3619, 0.8134, 0.2392],
            [0.8204, 0.2013, 0.9769,  ..., 0.7117, 0.0643, 0.4224],
            ...,
```

- ugh...

`aesthetify()` you tensors like this

```py
from aesthetic_tensor import aesthetify

aesthetify()  # monkey patch torch.Tensor

T.ae
```

```py
>> float32<3, 100, 100>∈[0.000, 1.000] | μ=0.499, σ=0.288
```

- much better

**you can also**

- **`T.ae.imshow()`**

```py
torch.rand(10, 10).ae.imshow()
```

![Random imshow](./assets/random-imshow.png)

- **`T.ae.hist()`**

```py
torch.rand(10, 10).ae.hist()
```

![Random hist](./assets/random-hist.png)

- **`T.ae.ploy()`**

```py
torch.rand(30).ae.plot(figsize=(6, 1))
```

![Random hist](./assets/random-plot.png)

- Check out the [docs](#) for more ways for visualizing.

## But my tensors are frequently batched

Calling the `.N` property on an `AestheticTensor` will give you an `AestheticCollection`,
pulling the leftmost dimension as a batch dimension.
You have the same interface as before, you just apply every transformation to every element.

An example will make everything much more clear.

```py
torch.rand(3, 2, 30).ae.N
```

```py
>>[3](~float32<2, 30>∈[0.032, 0.977] | μ=0.514, σ=0.293)
```

now you can choose how to plot each of the 3 elements of shape `<2, 30>`.
At the end, call the `.nb` (as in notebook) method to view the whole collection.

```py
torch.rand(3, 2, 30).ae.N.imshow(figsize=(6, 1)).nb(ncol=1)
```

![Random batched](assets/random-batched.png)

## Resources

- Check out [lovely-tensors](https://github.com/xl0/lovely-tensors) -  a similar library.
