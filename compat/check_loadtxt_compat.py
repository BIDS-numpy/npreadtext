
import argparse
import textwrap
import numpy as np
from npreadtext import _loadtxt


# Monkey patch numpy.loadtxt
np.loadtxt = _loadtxt


if __name__ == "__main__":
    descr = textwrap.dedent(
        """
        Run numpy tests with loadtxt replaced by a wrapper of the
        new reader.  Some useful tests to pass as the -t option:
            numpy.lib.tests.test_regression
            numpy.lib.tests.test_io::TestLoadTxt
        """)
    parser = argparse.ArgumentParser(
                description=descr,
                formatter_class=argparse.RawDescriptionHelpFormatter)

    test_help = ('Test to run, using the same syntax as the -t option '
                 'of runtests.py.')
    parser.add_argument('-t', '--test', required=True, help=test_help)

    verbose_help = ('Verbosity value for test outputs, in the range 1-3. '
                    'Default is 1.')
    parser.add_argument('-v', '--verbose', type=int, choices=[1, 2, 3],
                        default=1, help=verbose_help)
    ignore_help = (
        'comma-separated list of individual tests to ignore.\n\n'
        'For example: To run all the tests in TestLoadTxt *except* for '
        '`test_max_rows` and `test_1D`::\n\n'
        '  check_loadtxt_compat.py -t numpy.lib.tests.test_io::TestLoadTxt '
        '--ignore="test_max_rows,test_1D"\n'
    )
    parser.add_argument('-i', '--ignore', type=str, default='', help=ignore_help)

    args = parser.parse_args()

    # Convert --ignore str to extra_argv input of np.test via pytest -k flag
    extra_argv = None
    if args.ignore:
        extra_argv = [f'-k not {args.ignore.replace(",", " and not ")}']

    np.test(verbose=args.verbose, tests=[args.test], extra_argv=extra_argv)
