.. _create_components:


Build model components
######################


This section is useful mostly for users who want to create new models from scratch or customize existing models. Users who only want to run simulations from existing models may skip this section.

As an example, we will start with a simple model of phytoplankton growing on a single nutrient in a flow-through system. The model is based on the following equations:

.. math::

    \frac{d N}{d t}  =  d\,(N_0 - N) -\mu_{max}\,\left(\frac{N}{k_N + N}\right)\, P \\
    \frac{d P}{d t}  =  \mu_{max}\,\left(\frac{N}{k_N + N}\right)\,P - d\,P


It comprises two state variables, dissolved nutrients (:math:`N`) and phytoplankton (:math:`P`). The model expresses quantities in units of :math:`µM N` (i.e. :math:`\mu mol N m^{-3})`. The physical environment is a flow-through system corresponding to a laboratory chemostat setup. Growth medium with nutrient concentration :math:`N_0` [:math:`µM N`] flows into the system at a rate :math:`d`  [:math:`d^{-1}`]. The model components (:math:`N` & :math:`P`) flow out of the system at that same rate.

Phytoplankton growth :math:`\mu` (:math:`d^{-1}`) is described by `Monod kinetics <https://en.wikipedia.org/wiki/Monod_equation>`__.

.. math::

    \mu = \mu_{max}\,\left(\frac{N}{k + N}\right)

where :math:`k` [:math:`µM N`] is the half-saturation nutrient concentration, defined as the concentration at which half the maximum growth rate is achieved, :math:`N` is the ambient nutrient concentration, and :math:`\mu_{max}` [:math:`d^{-1}`] is the maximum growth rate achievable under ideal growth conditions.


We could just implement this numerical model with a few lines of Python / Numpy code. We will show, however, that it is very easy to refactor this model for using it with the XSO framework. We will also show that, while enabling useful features, the refactoring still results in a short amount of readable code.

The XSO framework allows structuring a model into modular components. The specific components we will use for this model are:

*   A component to define each of the state variables (:math:`N` and :math:`P`)
*   A component to define the external forcing (:math:`N_0`)
*   A component to define the linear inflow of nutrient into the system
*   A component to define the growth of phytoplankton on the nutrient
*   A component to define the linear outflow of the state variables from the system



Anatomy of a component
======================

A :class:`xso.component` is essentially a wrapper around a :class:`xsimlab:xsimlab.process` class, the main building block of the Xarray-simlab framework. There is a lot of additional functionality provided by ``xso`` through the *variable types* and the :deco:`xso.component` decorator. The following sections will explain the anatomy of a component via examples.

Component defining a variable
-----------------------------

Let's first write a generic component to define the state variables in our "NP chemostat" model. The *component* is a python class decorated by :deco:`xso.component`. Next we'll explain in detail the content of this class.

.. code-block:: python

    @xso.component
    class StateVariable:
        """XSO component to define a state variable in the model."""
        value = xso.variable(description='concentration of state variable',
                             attrs={'units': 'mmol N m-3'})


This simple component has only one attribute, which is an :func:`xso.variable` that defines a single state variable, ``value``, which represents the concentration of a state variable in our model. Metadata can be supplied via the ``description`` and ``attrs`` arguments. The ``attrs`` argument is a dictionary of attributes that will be added to the variable in the model dataset.


Component defining a flux
-------------------------

Next, we'll define a component that represents a flux in our model. In the XSO framework, a :deco:`xso.flux` is a decorated function within a component, that defines a part of the system of equations in our model. It represents the change of a state variable over time. Here we have a component to define the linear inflow of nutrient in our model:

.. code-block:: python

    @xso.component
    class LinearInflow:
        """Component defining the linear inflow of one variable."""
        sink = xso.variable(foreign=True, flux='input', negative=False)
        source = xso.forcing(foreign=True)
        rate = xso.parameter(description='linear rate of inflow')

        @xso.flux
        def input(self, sink, source, rate):
            return source * rate

The ``ìnput`` function has access to all *variable types* defined within the component. The link between a flux function and a variable is made via the ``flux`` argument of the :func:`xso.variable` attribute, which references the name of the flux function. The sign of the flux is defined by the ``negative`` argument, where ``negative=False`` is a flux adding to that variable, and ``negative=True`` is a flux subtracting from that variable.

The ``foreign=True``argument for sink and source allows passing the variable label at model setup. The same goes for the :func:`xso.forcing` that has to be initialized in another component and referenced via the forcing label.


The follwoing component can define the growth of our phytoplankton state variable on the nutrient:

.. code-block:: python

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


This component has four attributes, two :func:`xso.variable` and two :func:`xso.parameter`. The variables are here defined as referencing state variables in another component, via the ``foreign=True`` argument. Additionally, they have a flux defined.

The name of the flux function is referenced by the ``flux`` argument of the :func:`xso.variable`. The ``negative`` argument indicates whether the flux is positive or negative. In this case, the flux is negative for the resource and positive for the consumer, because it represents the uptake of nutrients by the phytoplankton. The flux function takes all *variable types* in the component as arguments, and defines an equation using standard python mathematical arguments.

Component defining a flux with dimension labels
------------------------------------------------

One of the more powerful features is the dimensionality and vectorization functionality built into python and leveraged by the framework. In this simple model, all state variables are non-dimensional. However we have two of the model components flowing out of the system. This could be implemented simply by adding two ``LinearOutflow`` components to our model, but we can also define a single component that handles the outflow of multiple variables:

.. code-block:: python

    @xso.component
    class LinearOutflow_ListInput:
        """Component defining the linear outflow of multiple variables."""
        var_list = xso.variable(dims='flow_list', list_input=True,
                               foreign=True, flux='decay', negative=True, description='variables flowing out')
        rate = xso.parameter(description='linear rate of outflow')

        @xso.flux(dims='flow_list')
        def decay(self, var_list, rate):
            # due to the list_input=True argument, var_list is an array of variables.
            # Thanks to vectorization we can just multiply the array with the rate.
            return var_list * rate


In this component, we define a single flux for multiple variables that are flowing out of the system. The ``list_input=True`` argument indicates that the variable labels can be supplied at *model setup* as a list. The XSO backend aggregates all labeled variables into an array, with the dimension label supplied via ``dims='flow_list'``. We need to make sure, that this dimension is also present in the ``dims`` argument of the :func:`xso.flux` decorator, so that the flux function knows how to handle the aggregated variables. The backend automatically routes the flux values to the appropriate variables, here as a negative function.

This allows for highly flexible complex model setups, as we can easily remove and add state variables, without overcomplicating our model structure.

The dimensionality not only allows using the ``list_input``argument, but also allows defining multi-dimensional state variables for our model. This is not used in this simple model, but would look like this:

.. code-block:: python

    @xso.component
    class StateVariableArray:
        """XSO component to define a state variable in the model."""
        values = xso.variable(dims='x', description='array of concentrations of state variables',
                             attrs={'units': 'mmol N m-3'})

Here the new dimension added to our model is labeled `'x'`, and this is a fixed label assigned to that specific component. Any other component or variable type referencing this variable has to provide a matching dimensionality. Dimension can be provided with meaningful metadata via the :func:`xso.index` variable type.

Component defining a forcing
----------------------------

.. code-block:: python

    @xso.component
    class ConstantExternalNutrient:
        """Component that provides a constant external nutrient
         as a forcing value.
        """

        forcing = xso.forcing(setup_func='forcing_setup', description='external nutrient')
        value = xso.parameter(description='constant value')

        def forcing_setup(self, value):
            """Method that returns forcing function providing the
            forcing value as a function of time."""
            @np.vectorize
            def forcing(time):
                return value

            return forcing

The component above is a simple constant forcing. The forcing could also be supplied via a parameter, but implementing it as a :func:`xso.forcing` with a forcing setup function allows exchanging this component with any other forcing function, e.g., with the following component defining a sinusoidal forcing:

.. code-block:: python

    @xso.component
    class SinusoidalExternalNutrient:
        """Component that provides a sinusoidal forcing value. """
        forcing = xso.forcing(setup_func='forcing_setup')
        period = xso.parameter(description='period of sinusoidal forcing')
        mean = xso.parameter(description='mean of sinusoidal forcing')
        amplitude = xso.parameter(description='amplitude of sinusoidal forcing')

      def forcing_setup(self, period, mean, amplitude):
            """Method that returns forcing function providing the
            forcing value as a function of time."""
            @np.vectorize
            def forcing(time):
                return mean + amplitude * self.m.sin(time / period * 2 * self.m.pi)

            return forcing


The forcing setup function is linked to the specific :func:`xso.forcing` variable via the ``setup_func`` argument. This needs to match the actual function name within the component. The setup function returns a vectorized function that provides the forcing value as a function of time (where time can be an array or a scalar), thus allowing the forcing to be compatible with any type of solver (i.e., with both fixed and adaptive step-size solvers).

.. note::

    The component above uses specific mathematical functions and variables, such as ``sin`` and ``pi``. To allow implementing a model with any solver backend, these are provided by that solver backend (available via the ``self.m`` attribute in any forcing or flux function).


Variable type options
~~~~~~~~~~~~~~~~~~~~~

Please see the following api reference, for a detailed overview of the available arguments to the *variable types* provided by the XSO framework:

.. autosummary::
    :toctree: _api_generated/

    xso.variable
    xso.parameter
    xso.forcing
    xso.flux
    xso.index

