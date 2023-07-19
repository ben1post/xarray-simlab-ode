Create, setup and run models
############################

From the pool of :doc:`components <workflow2_variables_components>` an :class:`xsimlab:xsimlab.Model` object can be created using the :func:`xso.create` function. With this *model object* we can create an input dataset for the model using :func:`xso.setup`. This input dataset is then used to setup and run the model. The output of the model run is stored in an output dataset. All of these datasets are essentially :class:`xarray:xarray.Dataset` objects, and can be readily used for further analysis and visualization.

Create a model
==============

A model is created by providing a dictionary of model *components* to the :func:`xso.create` function. The dictionary keys are the labels assigned to the model components within the model, and the values are the model component classes. The labels have to be unique, to identify instances of the same model components.

The model components can be defined in any order, and the model components can be defined in the same script as the model, or imported from other modules. The `xso` framework orders the components by execution order automatically.

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

    print(NPChemostat)

..  code-block:: console

    <xsimlab.Model (9 processes, 17 inputs)>
    Core
        solver_type       [in] solver type to use for model
    Time
        time_input        [in] ('time',) sequence of time for which to ...
    Nutrient
        value_label       [in] label / concentration of state variable
        value_init        [in] initial value / concentration of state v...
    Phytoplankton
        value_label       [in] label / concentration of state variable
        value_init        [in] initial value / concentration of state v...
    N0
        forcing_label     [in] label / external nutrient
        value             [in] parameter / constant value
    Inflow
        sink              [in] label reference /
        source            [in] label reference /
        rate              [in] parameter / linear rate of inflow
    Outflow
        var_list          [in] ('d',) label list / variables flowing out
        rate              [in] parameter / linear rate of outflow
    Growth
        resource          [in] label reference /
        consumer          [in] label reference /
        halfsat           [in] parameter / half-saturation constant
        mu_max            [in] parameter / maximum growth rate
    Solver

As can be seen from the ``__repr__`` of the model object, there are two perhaps unexpected aspects of the resulting model object:

#.   The number of variables does not exactly correspond to the number of variable types in the model. Each :func:`xso.variable` initialized within a *component* requires two input arguments. E.g., for the ``value`` variable, it is  ``value_label`` and ``value_init``. The ``value_label`` is the label of the variable, and the ``value_init`` is the initial value of the variable. The label supplied here can be used to reference the variable in another component within the same model (where the input parameter is a ``label reference``).
#. There are three model processes automatically added to the model object. These are the ``Solver``, ``Core`` and ``Time`` components provided by :mod:`xso.backendcomps`, that allow the model system to be setup and solved.

In order to setup the model with a custom unit for time (e.g. per month, per year) this can be supplied to the :func:`xso.create` function as the ``time_unit`` argument. The default unit for time is ``'d'`` for days.

Setup a model
=============

A model is setup by calling the :func:`xso.setup` function. The :func:`xso.setup` function takes the following arguments:

*   ``solver``: the solver to use for the model. Currently only ``solve_ivp`` (RK45 algorithm) and ``stepwise`` is supported.
*   ``model``: the *model object* to setup.
*   ``time``: the time array the model should be solved for.
*   ``input_vars``: a dictionary of input variables to use for the model. The dictionary keys are the model component labels, and the values are dictionaries of input variables for the model component and their required. The input variables are defined as follows.

..  code-block:: python

    chemostat_setup = xso.setup(solver='solve_ivp', model=NPChemostat,
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


The :func:`xso.setup` function returns an :class:`xarray:xarray.Dataset` with the input variables and model components as data variables.

..  code-block:: python

    print(chemostat_setup)


..  code-block:: console

    <xarray.Dataset>
    Dimensions:                     (clock: 2, d: 2, time: 1000)
    Coordinates:
      * clock                       (clock) float64 0.0 0.1
    Dimensions without coordinates: d, time
    Data variables: (12/17)
        Nutrient__value_label       <U1 'N'
        Nutrient__value_init        float64 1.0
        Phytoplankton__value_label  <U1 'P'
        Phytoplankton__value_init   float64 0.1
        Inflow__source              <U2 'N0'
        Inflow__rate                float64 0.1
        ...                          ...
        Growth__halfsat             float64 0.7
        Growth__mu_max              int64 1
        N0__forcing_label           <U2 'N0'
        N0__value                   float64 1.0
        Core__solver_type           <U9 'solve_ivp'
        Time__time_input            (time) float64 0.0 0.1 0.2 ... 99.7 99.8 99.9
    Attributes:
        __xsimlab_output_vars__:  Nutrient__value,Phytoplankton__value,Inflow__in...

The xarray-simlab backend automatically labels the variables within the dataset with the scheme (component label)__(variabel name), (e.g. ``N0__value``), ensuring a unique label for each variable.

Run a model
===========

Now that we have a *model object* and a corresponding *input dataset*, we can run the model.

When executing the model by calling the :meth:`.xsimlab.run() <xsimlab:xarray.Dataset.xsimlab.run>` method of the *model setup* and supplying the appropriate *model object*, a “filled-out” Xarray dataset is returned containing model setup parameters, metadata, and output.

..  code-block:: python

    chemostat_out = chemostat_setup.xsimlab.run(model=NPChemostat)

    print(chemostat_out)


..  code-block:: console

    <xarray.Dataset>
    Dimensions:                     (time: 1000, flow_list: 2, clock: 2)
    Coordinates:
      * clock                       (clock) float64 0.0 0.1
      * time                        (time) float64 0.0 0.1 0.2 ... 99.7 99.8 99.9
    Dimensions without coordinates: flow_list
    Data variables: (12/23)
        Core__solver_type           <U9 'solve_ivp'
        Growth__consumer            <U1 'P'
        Growth__halfsat             float64 0.7
        Growth__mu_max              int64 1
        Growth__resource            <U1 'N'
        Growth__uptake_value        (time) float64 0.06021 0.06021 ... 0.09224
        ...                          ...
        Outflow__rate               float64 0.1
        Outflow__var_list           (flow_list) <U1 'N' 'P'
        Phytoplankton__value        (time) float64 0.1 0.105 ... 0.9222 0.9222
        Phytoplankton__value_init   float64 0.1
        Phytoplankton__value_label  <U1 'P'
        Time__time_input            (time) float64 0.0 0.1 0.2 ... 99.7 99.8 99.9


The XSO framework currently provides two solver algorithms: an adaptive step-size solver from the SciPy package *solve_ivp* and a simple step-wise solver that is built into the backend Xarray-simlab framework.

Additional options to ``xsimlab.run()`` provided by Xarray-simlab are the ``batch_dim`` and ``parallel`` arguments.
The ``batch_dim`` argument is used to specify additional dimensions at which the model should be executed individually for each value in the dimension.
Currently, the ``parallel=True`` argument for model parallel execution along the ``batch_dim`` is only compatible with the ``stepwise`` solver provided by the XSO backend.


Storing input & output
======================

The simulation input and output is stored in the :class:`xarray:xarray.Dataset` structure. This structure already supports serialization and I/O to several file formats, where `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`__ is the recommended format. Please read Section `reading and writing files <http://xarray.pydata.org/en/stable/io.html>`__ in Xarray's docs, for more information.


..  code-block:: python

    chemostat_out.to_netcdf('output.nc')


For more details please read the `xarray-simlab documentation <https://xarray-simlab.readthedocs.io/en/latest/io_storage.html>`__.


Visualizing output
==================

The output of a model run can be visualized using the Xarray plotting functions :meth:`xarray:xarray.DataArray.plot`, which is available as a methods of the output dataset. The plotting functions are based on and fully compatible with the plotting library `matplotlib <https://matplotlib.org/>`__.

..  code-block:: python

    import matplotlib.pyplot as plt

    chemostat_out.Phytoplankton__value.plot(label='P')
    chemostat_out.Nutrient__value.plot(label='N')
    plt.ylabel(chemostat_out.Phytoplankton__value.attrs['units'])
    plt.legend()


..  image:: /_static/chemostatplot.png
    :class: with-shadow
