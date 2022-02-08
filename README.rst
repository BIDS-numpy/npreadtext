**NOTE:  This reader is now included in NumPy and is used for `np.loadtxt`.
The code in this repository is not up to date and the NumPy version should
be used.  There are known bugs or subtle differences only fixed in NumPy.**



npreadtext
==========

Read text files (e.g. CSV or other delimited files) into a NumPy array.

Quick Start
-----------

``npreadtext`` has been tested with NumPy v1.18 and higher and can be installed
using::

    python -m pip install numpy
    python -m pip install git+git://github.com/BIDS-numpy/npreadtext

To enable the C-accelerated version of ``np.loadtxt``, monkey-patch NumPy::

    import numpy as np
    from npreadtext import monkeypatch_numpy

This replaces ``np.loadtxt`` with ``npreadtext._loadtxt``.

For more detailed information on installation, testing, and benchmarking - see
below.

Dependencies
------------

Requires NumPy::

    pip install -r requirements.txt

To run the test and benchmarking suites, you will need some additional tools::

    pip install -r dev_requirements.txt

Build/Install
-------------

Build and install w/ pip: ``pip install -e .``. The ``--verbose`` flag is
useful for seing build logs: ``pip install -e . --verbose``.
Full (syntax-highlighted) build log also via ``python setup.py build_ext -i``.

Testing
-------

There are three sets of tests:

- npreadtxt test suite::

      pytest .

- Compatibility with ``np.loadtxt``::

      python compat/check_loadtxt_compat.py -t numpy.lib.tests.test_io::TestLoadTxt

Benchmarking
------------

The following is a quick-and-dirty procedure for evaluating the performance
of ``npreadtext`` with the numpy benchmark suite.
**TODO**: figure out how to get configure ``asv`` to do this comparison directly.
The pain point was getting ``npreadtext`` installed in the virtual environments
that ``asv`` creates.
This is a hacky procedure to work around these complications
by running everything in the same virtualenv and falling back on basic utils.

Setting up
~~~~~~~~~~

- Create new (empty) virtualenv
- In numpy repo:

  - ``pip install -r test_requirements.txt``
  - ``pip install -e .``
  - ``pip install asv virtualenv``

- In this repo:

  - ``pip install -e .``

- Back in numpy repo, create a branch (asv works best with committed changes):

  - ``git checkout -b monkeypatch-npreadtxt``
  - Modify the ``numpy/__init__.py`` to monkeypatch ``_loadtxt`` into numpy
    in place of ``np.loadtxt``. For example, delete the original loadtxt from
    ``__init__.py`` and modify the ``__getattr__`` to return ``_loadtxt``::

       del loadtxt
       def __getattr__(attr):
           if attr == "loadtxt":
               sys.path.append("/path/to/npreadtext/")
               from npreadtext import _loadtxt
               return _loadtxt
           ...

  - Commit the changes

Running
~~~~~~~

In the numpy repo, checkout the branch you want to compare against (presumably
``main``):

- ``git checkout main``
- ``python runtests.py --bench-compare monkeypatch-npreadtxt bench_io``

Comparing with other text loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is also a script ``bench/bench.py`` to facilitate basic performance
comparisons with other text loaders such as ``pd.read_csv``.
The script uses the IPython ``%timeit`` magic so should be run with ipython,
e.g.

.. code-block:: bash

   ipython -i bench/bench.py

Comparing with ``pandas``
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``pandas.read_csv`` uses an approximate method for parsing
floating point numbers. In practice, this results in faster float parsing
at the expense of faithful full-precision reproduction of floating point
values on reading/writing. Full-precision float parsing can be selected
using the ``float_precision="round-trip"`` option of ``pandas.read_csv``.

See also:

- https://pandas.pydata.org/docs/user_guide/io.html#specifying-method-for-floating-point-conversion
- https://github.com/pandas-dev/pandas/issues/17154
