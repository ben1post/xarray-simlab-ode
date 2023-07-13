Install xso
=====================

Required dependencies
---------------------

- Python 3.6 or later.
- [attrs](http://www.attrs.org) (18.2.0 or later)
- [xarray](http://xarray.pydata.org) (0.10.0 or later)
- [zarr](https://zarr.readthedocs.io) (2.3.0 or later)
- [dask](https://docs.dask.org) (0.18.0 or later)
- [scipy](https://www.scipy.org) (1.1.0 or later)
- [numpy](https://www.numpy.org) (1.15.0 or later)
- [xarray-simlab](https://xarray-simlab.readthedocs.io) (0.5.0 or later)

Optional dependencies
---------------------

- [graphviz](http://graphviz.readthedocs.io) (for model visualization)
- [tqdm](https://tqdm.github.io) (for progress bars)

Install using pip
-----------------

The easiest way to install xarray-simlab-ode and its required dependencies is using
``pip``:

```console
pip install xso
```

Install using conda
-------------------

Currently, xarray-simlab-ode is not available via conda. However, you can 
install all required dependencies by installing xarray-simlab and scipy using conda:

```console
conda install -c conda-forge xarray-simlab scipy
```

Then just install xarray-simlab-ode in the same environment using pip:

```console
pip install xso
```


Install from source
-------------------

To install xarray-simlab-ode from source 
(i.e. the actively developed version on [GitHub](https://github.com/ben1post/xarray-simlab-ode)),
be sure you have the required
dependencies (numpy and xarray) installed first. You might consider
using conda to install them:

```console 
conda install attrs xarray dask zarr numpy scipy xarray-simlab pip -c conda-forge
```

A good practice (especially for development purpose) is to install the
packages in a separate environment, e.g. using conda:

```console
conda create -n xso_env python attrs xarray dask zarr numpy scipy xarray-simlab pip -c conda-forge

source activate xso_env
```

Then you can clone the xarray-simlab git repository and install it
using ``pip`` locally:
```console
git clone https://github.com/ben1post/xarray-simlab-ode
cd xarray-simlab-ode
pip install .
```

For development purpose, use the following command:

```console
pip install -e .
```


Import xarray-simlab-ode
--------------------

To make sure that xarray-simlab-ode is correctly installed, try to import
it by running this line:

    $ python -c "import xso"