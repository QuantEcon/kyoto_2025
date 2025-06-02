---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Fun with JAX

#### [John Stachurski](https://johnstachurski.net/) 

June 2025

This notebook illustrates the power of [JAX](https://github.com/google/jax), a Python library built by Google Research.

It should be run on a machine with a GPU --- for example, try Google Colab with the runtime environment set to include a GPU.

The aim is just to give a small taste of high performance computing in Python -- details will be covered later in the course.


We start with some imports

```python
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
```

Let's check our hardware:

```python
!nvidia-smi
```

```python
!lscpu -e
```

## Transforming Data


A very common numerical task is to apply a transformation to a set of data points.

Our transformation will be the cosine function.


Here we evaluate the cosine function at 50 points.

```python
x = np.linspace(0, 10, 50)
y = np.cos(x)
```

Let's plot.

```python
fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()
```

Our aim is to evaluate the cosine function at many points.

```python
n = 50_000_000
x = np.linspace(0, 10, n)
```

### With NumPy

```python
%time np.cos(x)
```

```python
%time np.cos(x)
```

```python
x = None  
```

### With JAX

```python
x_jax = jnp.linspace(0, 10, n)
```

```python
jnp.cos(x_jax)
```

Let's time it.

(The `block_until_ready()` method is only needed for timing.)

```python
%time jnp.cos(x_jax).block_until_ready()
```

```python
%time jnp.cos(x_jax).block_until_ready()
```

Here we change the input size --- can you explain why the timing changes?

```python
x_jax = jnp.linspace(0, 10, n + 1)
```

```python
%time jnp.cos(x_jax).block_until_ready()
```

```python
%time jnp.cos(x_jax).block_until_ready()
```

```python
x_jax = None  # Free memory
```

## Evaluating a more complicated function

```python
def f(x):
    y = np.cos(2 * x**2) + np.sqrt(np.abs(x)) + 2 * np.sin(x**4) - 0.1 * x**2
    return y
```

```python
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, f(x))
ax.scatter(x, f(x))
plt.show()
```

Now let's try with a large array.


### With NumPy

```python
n = 50_000_000
x = np.linspace(0, 10, n)
```

```python
%time f(x)
```

```python
%time f(x)
```

### With JAX

```python
def f(x):
    y = jnp.cos(2 * x**2) + jnp.sqrt(jnp.abs(x)) + 2 * jnp.sin(x**4) - x**2
    return y
```

```python
x_jax = jnp.linspace(0, 10, n)
```

```python
%time f(x_jax).block_until_ready()
```

```python
%time f(x_jax).block_until_ready()
```

### Compiling the Whole Function

```python
f_jax = jax.jit(f)
```

```python
%time f_jax(x_jax).block_until_ready()
```

```python
%time f_jax(x_jax).block_until_ready()
```

## More

See [the QuantEcon JAX lectures](https://jax.quantecon.org/intro.html).


