# Welcome to xarray-simlab-ode's documentation!

Xarray-simlab-ode provides the `xso` framework for building and solving models based on ordinary differential 
equations (ODEs), an extension of [xarray-simlab](https://github.com/xarray-contrib/xarray-simlab).

Xarray-simlab provides a generic framework for building computational models in a modular fashion 
and a [xarray](http://xarray.pydata.org/) extension for setting and running simulations using xarray's `Dataset` 
structure.

Xarray-simlab-ode (XSO) extends the Xarray-simlab framework with a set of variables, processes and a solver backend, 
suited towards ODE-based models. It is designed for flexible, interactive and reproducible modeling workflows.

*Note: This project is in the early stages of development.*

## Documentation table of contents


```{toctree}
---
caption: Getting Started
maxdepth: 1
---

about
install

```

```{toctree}
---
caption: User Guide
maxdepth: 1
---

workflow1_framework
workflow2_variables_components
workflow3_models

```


```{toctree}
---
caption: Help & Reference
maxdepth: 1
---

api
contributing
citation
changelog
```

## Get involved

The package is in the early stages of development. Feedback from testing and contributions are very welcome. 
See [GitHub Issues](https://github.com/ben1post/xarray-simlab-ode/issues) for existing issues, or raise your own.
Code contributions can be made via Pull Requests on [GitHub](https://github.com/ben1post/xarray-simlab-ode).
Check out the [contributing guidelines](contributing) for more information.

## License

xarray-simlab-ode was created by Benjamin Post. 
It is licensed under the terms of the BSD 3-Clause license.

## Credits

Xarray-simlab-ode is an extension of [xarray-simlab](https://github.com/xarray-contrib/xarray-simlab), created by Benoît Bovy.
