# xarray-simlab-ode

The`xso`Framework for building and solving ODE-based models, an extension of xarray-simlab.

## Installation

```bash
$ pip install xso
```

## Usage
A highly simplified model based on ordinary differential equations is shown below.
1. Create Create new model components by writing compact Python classes:
```python
import xso

@xso.component
class Variable:
    var = xso.variable(description='basic state variable')

@xso.component
class LinearGrowth:
    var_ext = xso.variable(foreign=True, flux='growth', description='external state variable')
    rate = xso.parameter(description='linear growth rate')
    @xso.flux
    def growth(self, var_ext, rate):
        return var_ext * rate
```
2. Create a new model just by providing a dictionary of model components:

```python
model = xso.create({'Var':Variable,'Growth':LinearGrowth})
```
3. Create an input xarray.Dataset, run the model and get an output xarray.Dataset:

```python
input_ds = xso.setup(solver='stepwise',
                     model=model,
                     input_vars={
                         'Var':{'var_label':'X', 'var_init':1},
                         'Growth':{'var':'A', 'rate':1.},
                     })

with model:
    output_ds = input_ds.xsimlab.run()
```
4.Perform model setup, pre-processing, run, post-processing and visualization in a functional style, using method chaining:
```python
'xxx'
```

## Contributing

The package is in the early stages of development, and contributions would be very welcome. See GitHub Issues for specific issues, or raise your own.
Code contributions can be made via Pull Requests on GitHub.
Check out the contributing guidelines for more specific information.

## License

xarray-simlab-ode was created by Benjamin Post. 
It is licensed under the terms of the BSD 3-Clause license.
- licences of dependencies?

## Credits

Xarray-simlab-ode is an extension of [xarray-simlab](https://github.com/xarray-contrib/xarray-simlab), created by Benoît Bovy.
