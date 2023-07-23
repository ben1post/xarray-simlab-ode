Introduction to the framework
##############################

The Xarray-simlab-ODE (``xso``) is a Python framework that allows users to construct and customize, in a modular fashion, models based on ordinary differential equations (ODEs). It is technically a wrapper around the modeling framework `xarray-simlab <https://xarray-simlab.readthedocs.io/en/latest/>`__, additionally providing a functional set of variables, processes and a solver backend for ODE-based models.

It is a non-opinionated framework, i.e., it does not provide a fixed notion of how the model should be structured, instead it attempts to remove the redundant boilerplate code, allowing a user to construct and work with ODE-based models.

XSO was developed as the technical foundation of the `Phydra library <https://github.com/ben1post/phydra>`__ for marine ecosystem modeling, but is not limited to any particular domain and can be used to create ODE-based models of any type.


Building blocks
===============

The XSO framework is built around the following concepts:

* Model data (setup and output)
* Model objects
* Components
* Variable types

The XSO framework provides several *variable types*, which directly correspond state variables, parameters, forcing, and mathematical terms (here called *fluxes*), that together build the system of equations.

Model *components* can be constructed as compact Python classes from the provided set of *variable types* and wrap a logical component of the model as the user sees fit.

One or more of these *components* can then be assembled to a *model object*, which defines the model structure. Components can interact with each other via the *variable types* and the labels supplied at *model setup*.

The *model setup* dataset contains all relevant information needed at runtime, such as the solver algorithm to be used, as well as time steps and model parameterization. State variables, forcing and parameters need to be initialized in one *component*, but can be referenced across the model (using the :code:`foreign=True` argument). The system of differential equations is constructed from the *fluxes* contained in the model *components* via the supplied labels at model setup.

With both the *model object* and the corresponding *model setup*, the model can be executed. Both *model setup* and the output is returned as an Xarray dataset with all metadata, which can be easily stored and shared.


Variable types
______________

*Variable types* are the most granular elements of the framework, which directly correspond to the basic mathematical components of ODE-based models (e.g., state variables, parameters, forcing, and partial equations). XSO currently provides the following *variable types*:

- :func:`xso.variable`: Defines a state variable in a *component*, either locally or via reference in another *component*.

- :func:`xso.forcing`: Defines an external forcing as a constant or time-varying value, via an additional setup function. Can also be a reference to a forcing in another *component*.

- :func:`xso.parameter`: Defines a constant model parameter within the *component*.

- :func:`xso.flux`: Defines a mathematical term with the *variable types* in the *component*, and adds the term into the system of differential equations of the underlying model. The flux function decorator provides a group argument, that allows linking multiple fluxes flexibly between components.

- :func:`xso.index`: Creates an input parameter to define a labelled dimension (i.e., Xarray index) within the model, stored as metadata in the model and output dataset.

These can be used to define variables in compact Python classes, to construct functional XSO *components*. All of them can be defined with a variable number of dimensions (i.e., as a vector, array, or matrix).

Components
__________

*Components* are the general building-blocks of a model that declare a subset of variables and define a specific set of mathematical functions computed for these variables during model runtime. More specifically, a *component* refers to a Python class containing XSO *variable types* that is decorated with :deco:`xso.component`.

For example, a *component* could define a specific nutrient uptake function, e.g. Monod-type phytoplankton growth on a single nutrient. The decorating function registers the *variable types* within the framework, reducing boilerplate code and creating fully functional model building blocks. *Components* can be reused within a model.

..  code-block:: python

    @xso.component
    class MonodGrowth:
        """Component defining a growth process based on Monod-kinetics."""
        resource = xso.variable(foreign=True, flux='uptake', negative=True)
        consumer = xso.variable(foreign=True, flux='uptake', negative=False)

        halfsat = xso.parameter(description='half-saturation constant')
        mu_max = xso.parameter(description='maximum growth rate')

        @xso.flux
        def uptake(self, mu_max, resource, consumer, halfsat):
            return mu_max * resource / (resource + halfsat) * consumer


   
Model object
============

A *Model object* is an instance of the Model class provided by Xarray-simlab. They consist of an ordered, immutable collection of *components*. A XSO *model object* is created with a call to the function :func:`xso.create()` by supplying a dictionary of model *components* with their respective labels. *Model objects* contain the *components* relevant to a model and can be easily stored and shared. They do not contain custom parameterization.

..  code-block:: python

    NPChemostat = xso.create({
        # State variables
        'Nutrient': StateVariable,
        'Phytoplankton': StateVariable,

        # Flows:
        'Inflow': LinearInflow,
        'Outflow': LinearOutflow_ListInput,

        # Growth
        'Growth': MonodGrowth,

        # Forcings
        'N0': ConstantExternalNutrient
    })


Model setup
===========

A *Model setup* is a Xarray dataset, that includes all relevant information needed at runtime, such as the *model object*, solver algorithm to be used, as well as time steps and model parameterization. A XSO *model setup* is created with a call to the function :func:`xso.setup` and supplying the aforementioned information as arguments. At this step, the *variable types* initialized in a *component* must be supplied with a value, as well as a label that can be used to reference them in other *components*. The model parameterization is passed as a dictionary, with the *component* labels used to create the *model object* as keys.

..  code-block:: python

    chemostat_setup = xso.setup(solver='stepwise', model=NPChemostat,
                time=np.arange(0,100, 0.1),
                input_vars={
                        # State variables
                        'Nutrient':{'value_label':'N','value_init':1.},
                        'Phytoplankton':{'value_label':'P','value_init':0.1},

                        # Flows:
                        'Inflow':{'source':'N0', 'rate':0.1, 'sink':'N'},
                        'Outflow':{'var_list':['N', 'P'], 'rate':0.1},

                        # Growth
                        'Growth':{'resource':'N', 'consumer':'P', 'halfsat':0.7, 'mu_max':1},

                        # Forcings
                        'N0':{'forcing_label':'N0', 'value':1.}
                })


Model execution
===============

The system of differential equations is constructed from the *fluxes* using the labels supplied during model setup. The number of values in a defined dimension is flexible, but they have to match across the model in order for the model to run.

When executing the model by calling the :meth:`.xsimlab.run() <xsimlab:xarray.Dataset.xsimlab.run>` method of the *model setup* and supplying the appropriate *model object*, a “filled-out” Xarray dataset is returned containing model setup parameters, metadata, and output.

.. code-block:: python

    chemostat_out = chemostat_setup.xsimlab.run(model=NPChemostat)


The XSO framework currently provides two solver algorithms: an adaptive step-size solver from the SciPy package *solve_ivp* and a simple step-wise solver that is built into the backend Xarray-simlab framework. Apart from the technical limitations of the solver algorithm used, there are no restrictions on the dimensionality and number of *variable types* used within a *component* and no limitations to the levels of *group* variables linking components to define a single ecosystem process.
