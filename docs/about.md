# About xarray-simlab-ode

Xarray-simlab-ode provides the `xso` framework for building and solving models based on ordinary differential equations 
(ODEs), an extension of [xarray-simlab](https://github.com/xarray-contrib/xarray-simlab).

Xarray-simlab provides a generic framework for building computational models in a modular fashion and an [xarray](http://xarray.pydata.org/) extension for setting and running simulations using xarray's `Dataset` structure.

Xarray-simlab-ode (XSO) extends the Xarray-simlab framework with a set of variables, processes and a solver backend, 
suited towards ODE-based models. It is designed for flexible, interactive and reproducible modeling workflows.

The `xso` framework was developed in conjunction with and provides the foundation for the [Phydra library]
(https://github.com/ben1post/phydra) of marine plankton community models.

## Installation

```bash
$ pip install xso
```

## In a nutshell

A highly simplified model based on ordinary differential equations is shown below.
1. Create new model components by writing compact Python classes:
```python
import xso

@xso.component
class Variable:
    var = xso.variable(description='basic state variable', attrs={'units':'µM'})

@xso.component
class LinearGrowth:
    var_ext = xso.variable(foreign=True, flux='growth', description='external state variable')
    rate = xso.parameter(description='linear growth rate', attrs={'units':'$d^{-1}$'})

    @xso.flux
    def growth(self, var_ext, rate):
        return var_ext * rate
```
2. Create a new model just by providing a dictionary of model components:

```python
model = xso.create({'Var':Variable,'Growth':LinearGrowth}, time_unit='d')
```
3. Create an input xarray.Dataset, run the model and get an output xarray.Dataset:

```python
import numpy as np

input_ds = xso.setup(solver='solve_ivp',
                     model=model,
                     time=np.arange(1,10,.1),
                     input_vars={
                         'Var':{'value_label':'X', 'value_init':1},
                         'Growth':{'var_ext':'X', 'rate':1.},
                     })

with model:
    output_ds = input_ds.xsimlab.run()
```
4.Perform model setup, pre-processing, run, post-processing and visualization in a functional style, using method chaining:
```python
with model:
    batchout_ds = (input_ds
     .xsimlab.update_vars(
         input_vars={'Growth': {'rate': ('batch', [0.9, 1.0, 1.1, 1.2])}}
     )
     .xsimlab.run(parallel=True, batch_dim='batch')
     .swap_dims({'batch':'Growth__rate'})
     .Var__var_value.plot.line(x='time')
     )
```

![plot](_static/GrowthRate_BatchOut.png)

## Motivation

Heavily borrowing in both function and design from [xarray-simlab](https://xarray-simlab.readthedocs.io/en/latest/), 
xarray-simlab-ode (XSO) provides a framework for building and solving models based on ordinary differential 
equations (ODEs). 

It is designed for flexible, interactive and reproducible modeling workflows, where the model does 
not get increasingly inflexible and difficult to reconfigure and maintain as it is developed. Additionally, XSO can 
provide an integrated modeling environment within the Python scientific ecosystem, with direct compatibility to 
tools for data analysis and visualization.

The goal is to contribute to the ongoing efforts of developing more robust, transparent, and reproducible models, 
moving away from monolithic and inflexible codes to a model development process that is inherently collaborative.

Scientists working with computational models do not always build the models themselves. Often, scientists use 
existing models and focus the work on parameterization and analysis of results obtained with model applications in 
specific locations. A user can start working with models without detailed knowledge of the underlying framework and 
learn the basic workflow before progressing to building custom models using the XSO framework. Additionally, more 
advanced users can easily share custom _components_ or _model objects_ via the respective Python objects. 

