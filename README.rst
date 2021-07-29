npreadtext
==========

Read text files (e.g. CSV or other delimited files) into a NumPy array.

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

- C-tests::

      cd src/ctests && source build_runtest.sh
      ./runtests

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
  - Modify the ``bench_io.py`` benchmark file to patch loadtxt, e.g. at the top
  of the file::
    
      from npreadtext._loadtxt import _loadtxt
      np.loadtxt = _loadtxt

  - Commit the changes

Running
~~~~~~~

In the numpy repo, checkout the branch you want to compare against (presumably
``main``):

- ``git checkout main``
- ``cd benchmarks``
- ``asv run -n -e --python=same -b bench_io |tee > /tmp/main.bench``

Then run the same procedure on the patched branch:

- ``git checkout monkeypatch-npreadtext``
- ``asv run -n -e --python=same -b bench_io |tee > /tmp/npreadtext.bench``

The results can be compared simply with ``diff``::

    diff -y --color /tmp/master.bench /tmp/npreadtext.bench

Comparing with other text loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is also a script ``bench/bench.py`` to facilitate basic performance
comparisons with other text loaders such as ``pd.read_csv``.
The script uses the IPython ``%timeit`` magic so should be run with ipython,
e.g.

.. code-block:: bash

   ipython -i bench/bench.py
