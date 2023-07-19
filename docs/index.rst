Welcome to xarray-simlab-ode's documentation!
#############################################

Xarray-simlab-ode provides the `xso` framework for building and solving models based on ordinary differential 
equations (ODEs), an extension of `xarray-simlab <https://github.com/xarray-contrib/xarray-simlab>`__.

Xarray-simlab provides a generic framework for building computational models in a modular fashion 
and a `xarray <http://xarray.pydata.org/>`__ extension for setting and running simulations using xarray's ``Dataset``
structure.

Xarray-simlab-ode (XSO) extends the Xarray-simlab framework with a set of variables, processes and a solver backend, 
suited towards ODE-based models. It is designed for flexible, interactive and reproducible modeling workflows.

.. Note:: This project is in the early stages of development.

Documentation table of contents
===============================

** Getting Started **

* :doc:`about`
* :doc:`install`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    about
    install

** User Guide **

* :doc:`workflow1_framework`
* :doc:`workflow2_variables_components`
* :doc:`workflow3_models`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User Guide

    workflow1_framework
    workflow2_variables_components
    workflow3_models


** Help & Reference **
* :doc:`api`
* :doc:`contributing`
* :doc:`citation`
* :doc:`changelog`


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & Reference

    api
    contributing
    citation
    changelog


Get involved
============

The package is in the early stages of development. Feedback from testing and contributions are very welcome. 
See [GitHub Issues](https://github.com/ben1post/xarray-simlab-ode/issues) for existing issues, or raise your own.
Code contributions can be made via Pull Requests on [GitHub](https://github.com/ben1post/xarray-simlab-ode).
Check out the [contributing guidelines](contributing) for more information.

License
=======

xarray-simlab-ode was created by Benjamin Post. 
It is licensed open source under the terms of the BSD 3-Clause license.

Credits
=======

Xarray-simlab-ode is an extension of [xarray-simlab](https://github.com/xarray-contrib/xarray-simlab), created by Benoît Bovy.
