# Build model components

## Design principles
Our goal in developing the framework was to allow users to build models without restricting the level of complexity,
in particular in relation to the dimensionality, number of state variables and model processes. 

This was implemented in the framework by providing _variable types_, which directly correspond to the basic 
mathematical components of models based on ordinary differential equations (e.g., state variables, parameters, 
forcing, and 
partial equations). 

Every aspect of the model needs to be defined at the level of _variable types_. Model 
_components_ can be flexibly constructed from the provided set of _variable types_ and wrap a logical 
component of the model as users see fit. 

State variables, forcing and parameters need to be initialized in one 
_component_, but can be referenced across the model. The system of differential equations is constructed 
from the _fluxes_ contained in the model _components_ via the supplied labels at model setup. 

To provide a template for utilizing this flexible framework, we present fully implemented marine ecosystem models
in the [Phydra library](https://github.com/ben1post/phydra).


## Variable types

### State variables

State variables are the core of the model and represent the quantities that are changing over time. They are
defined by a name, a description, and a unit. The unit is optional, but recommended. The state variables are
defined as a Python class, which inherits from the `xso.Variable` class. The state variable class can be
decorated with the `@xso.variable` decorator.


### Parameters

Parameters are quantities that are constant over time. They are defined by a name, a description, and a unit.
The unit is optional, but recommended. The parameters are defined as a Python class, which inherits from the
`xso.Parameter` class. The parameter class can be decorated with the `@xso.parameter` decorator. The decorator
takes the following arguments:

- `description`: A string describing the parameter.
- `attrs`: A dictionary of attributes that will be added to the parameter in the model dataset.
- `unit`: A string describing the unit of the parameter. The unit is optional, but recommended.
- `value_label`: A string describing the label of the parameter in the model dataset. If not provided, the
  name of the parameter will be used.
- `value`: A float or array of floats describing the value of the parameter. If not provided, the parameter
  will be initialized with zeros.

### Forcing

Forcing is a quantity that is constant over time, but varies across space. They are defined by a name, a
description, and a unit. The unit is optional, but recommended. The forcing are defined as a Python class,
which inherits from the `xso.Forcing` class. The forcing class can be decorated with the `@xso.forcing`
decorator. The decorator takes the following arguments:
    
- `description`: A string describing the forcing.


### Fluxes

Fluxes are quantities that are changing over time and vary across space. They are defined by a name, a
description, and a unit. The unit is optional, but recommended. The fluxes are defined as a Python class,
which inherits from the `xso.Flux` class. The flux class can be decorated with the `@xso.flux` decorator.
The decorator takes the following arguments:

- `description`: A string describing the flux.


## Components

Components are the building blocks of the model. They are defined by a name, a description, and a list of
variables. The variables can be state variables, parameters, forcing, or fluxes. The components are defined
as a Python class, which inherits from the `xso.Component` class. 

The component class can be decorated with the `@xso.component` decorator. 




## Model

The model is the top-level object that contains all the components and variables. It is defined by a name, a
description, and a list of components. The components can be state variables, parameters, forcing, or fluxes.


`xso.create()` takes the following arguments:

