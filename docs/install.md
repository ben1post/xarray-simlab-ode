.. _installing:

Install xarray-simlab-ode
=====================

Required dependencies
---------------------

- Python 3.6 or later.
- `attrs <http://www.attrs.org>`__ (18.2.0 or later)
- `numpy <http://www.numpy.org/>`__
- `scipy <http://www.scipy.org/>`__
- `xarray <http://xarray.pydata.org>`__ (0.10.0 or later)
- `zarr <https://zarr.readthedocs.io>`__ (2.3.0 or later)
- `dask <https://docs.dask.org>`__
- `xarray-simlab <https://xarray-simlab.readthedocs.io>`__ (0.5.0 or later) 

Optional dependencies
---------------------

- `graphviz <http://graphviz.readthedocs.io>`__ (for model visualization)
- `tqdm <https://tqdm.github.io>`__ (for progress bars)

Install using pip
-----------------

You can also install xarray-simlab-ode and its required dependencies using
``pip``:

```console
pip install xso
```

Install using conda
-------------------

Currently xarray-simlab-ode is not available via conda. However, you can 
install all required dependencies by installing xarray-simlab using conda, 
and then just install xarray-simlab-ode in the same environment using pip:

```console
conda install xarray-simlab -c conda-forge

pip install xso
```


Install from source
-------------------

To install xarray-simlab-ode from source, be sure you have the required
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

.. _PyPi: https://pypi.python.org/pypi/xso/

Import xarray-simlab-ode
--------------------

To make sure that xarray-simlab-ode is correctly installed, try to import
it by running this line:

    $ python -c "import xso"