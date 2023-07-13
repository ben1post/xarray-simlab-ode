# Create, setup and run models

From the pool of [model components](workflow_variables_components), themselves built up of `xso` variable 
types, a model can be created.

## Create a model

A model is created by providing a dictionary of model components to the `xso.create` function. The dictionary keys 
are the labels of the model components within the model, and the values are the model component classes. The model 
components are defined as Python classes using the `@xso.component` decorator. 
The model components can be defined in the same script as the model, or imported from other modules.

The model components can be defined in any order, and the model components can be defined in the same script as the
model, or imported from other modules. The `xso` framework orders the components by execution order automatically.


```python
import xso

model = xso.create({'Var':Variable,'Growth':LinearGrowth}, time_unit='d')

print(model)
```

```console
<xsimlab.Model (5 processes, 6 inputs)>
Core
    solver_type     [in] solver type to use for model
Time
    time_input      [in] ('time',) sequence of time for which to so...
Var
    value_label     [in] label / basic state variable
    value_init      [in] initial value / basic state variable
Growth
    var_ext         [in] label reference / external state variable
    rate            [in] parameter / linear growth rate
Solver
```

## Setup a model

A model is setup by calling the `xso.setup` function. The `xso.setup` function takes the following arguments:
- `solver`: the solver to use for the model. Currently only `solve_ivp` (RK45 algorithm) and `stepwise` is supported.
- `model`: the _model object_ to setup.
- `time`: the time array the model should be solved for.
- `input_vars`: a dictionary of input variables to use for the model. The dictionary keys are the model component  
  labels, and the values are dictionaries of input variables for the model component and their required. The input 
  variables are 
  defined as follows.

```python
import numpy as np

input_ds = xso.setup(solver='solve_ivp',
                     model=model,
                     time=np.arange(1,10,.1),
                     input_vars={
                         'Var':{'value_label':'X', 'value_init':1},
                         'Growth':{'var_ext':'X', 'rate':1.},
                     })

```
The `xso.setup` function returns an `xarray.Dataset` with the input variables and model components as data variables.

```python
print(input_ds)
```

```console
<xarray.Dataset>
Dimensions:            (clock: 2, time: 90)
Coordinates:
  * clock              (clock) float64 1.0 1.1
Dimensions without coordinates: time
Data variables:
    Var__value_label   <U1 'X'
    Var__value_init    int64 1
    Growth__var_ext    <U1 'X'
    Growth__rate       float64 1.0
    Core__solver_type  <U9 'solve_ivp'
    Time__time_input   (time) float64 1.0 1.1 1.2 1.3 1.4 ... 9.6 9.7 9.8 9.9
Attributes:
    __xsimlab_output_vars__:  Var__value,Growth__growth_value
```

## Run a model

A model is run by calling the `xsimlab.run` method on the input dataset. 

- mention context options 
  - with ProgressBar
  - with Model
- batch dimension
- 
```python
with model:
    output_ds = input_ds.xsimlab.run()

print(output_ds)
```

```console
<xarray.Dataset>
Dimensions:               (time: 90, clock: 2)
Coordinates:
  * clock                 (clock) float64 1.0 1.1
  * time                  (time) float64 0.0 0.1 0.2 0.3 0.4 ... 8.6 8.7 8.8 8.9
Data variables:
    Core__solver_type     <U9 'solve_ivp'
    Growth__growth_value  (time) float64 1.052 1.052 ... 6.313e+03 6.969e+03
    Growth__rate          float64 1.0
    Growth__var_ext       <U1 'X'
    Time__time_input      (time) float64 1.0 1.1 1.2 1.3 1.4 ... 9.6 9.7 9.8 9.9
    Var__value            (time) float64 1.0 1.105 1.221 ... 6.626e+03 7.323e+03
    Var__value_init       int64 1
    Var__value_label      <U1 'X'
```



## Storing input & output

The input and output of a model run is stored in an `xarray.Dataset`.


```python
output_ds.to_netcdf('output.nc')
```

For more details please read 
the [xarray-simlab documentation](https://xarray-simlab.readthedocs.io/en/latest/io_storage.html). There are 

## Visualizing output

The output of a model run can be visualized using the `xarray` plotting functions, and are available as methods on 
the output dataset.

```python
output_ds.Var__value.plot()
```


