.. _api:

##################
API Overview
##################

This page points to the most important classes and functions of the
xarray-simlab-ode API.


Top-level functions
===================

.. currentmodule:: xso
.. autosummary::
    :toctree: _api_generated/

    xso.create
    xso.setup


Component
=========

.. autosummary::
    :toctree: _api_generated/

    xso.component


Variable types
==============

.. autosummary::
    :toctree: _api_generated/

    xso.variable
    xso.parameter
    xso.forcing
    xso.flux
    xso.index


Backend components
==================

.. autosummary::
    :toctree: _api_generated/

    xso.core.XSOCore
    xso.model.Model
    xso.solvers.SolverABC
    xso.solvers.IVPSolver
    xso.solvers.StepwiseSolver
    xso.backendcomps.Backend
    xso.backendcomps.Context
    xso.backendcomps.Time
    xso.backendcomps.RunSolver



Xarray-simlab
=================================

Since XSO mostly wraps `xarray-simlab <https://xarray-simlab.readthedocs.io/en/latest/>`_, it can be helpful to also reference that API.

* The :deco:`xso.component`decorator function returns a fully functional :deco:`xsimlab:xsimlab.process` class.
* The :func:`xso.create` function returns a :class:`xsimlab:xsimlab.Model` instance.
* The :func:`xso.setup` function is a light wrapper around :func:`xsimlab:xsimlab.create_setup`.
* All *variable types* are represented in the model as :func:`xsimlab:xsimlab.variable` instances.

We can access the xarray-simlab API through the ``xsimlab`` accessor of the created input and output xarray Datasets:

.. code-block:: python

    >>> import xarray as xr         # first import xarray
    >>> import xsimlab              # import xsimlab (the 'xsimlab' accessor is registered)
    >>> ds = xr.Dataset()           # create or load an xarray Dataset
    >>> ds.xsimlab.<meth_or_prop>   # access to the methods and properties listed below

.. currentmodule:: xarray

