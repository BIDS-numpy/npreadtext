
import operator
import numpy as np
from . import _flatten_dtype
from ._readtextmodule import (_readtext_from_filename,
                              _readtext_from_file_object)


def read(file, *, delimiter=',', comment='#', quote='"',
         decimal='.', sci='E', usecols=None, skiprows=0, dtype=None):
    """
    Read a NumPy array from a text file.

    Parameters
    ----------
    file : str or file object
        The filename or the file to be read.
    delimiter : str, optional
        Field delimiter of the fields in line of the file.
        Default is a comma, ','.
    comment : str, optional
        Character that begins a comment.  All text from the comment
        character to the end of the line is ignored.
    quote : str, optional
        Character that is used to quote string fields. Default is '"'
        (a double quote).
    decimal : str, optional
        The decimal point character.  Default is '.'.
    sci : str, optional
        The character in front of the exponent when exponential notation
        is used for floating point values.  The default is 'E'.  The value
        is case-insensitive.
    usecols : array_like, optional
        A one-dimensional array of integer column numbers.  These are the
        columns from the file to be included in the array.  If this value
        is not given, all the columns are used.
    skiprows : int, optional
        Number of lines to skip before interpreting the data in the file.
    dtype : numpy data type, optional
        If not given, the data type is inferred from the values found
        in the file.

    Returns
    -------
    ndarray
        NumPy array.

    Examples
    --------
    First we create a file for the example.

    >>> s1 = '1.0,2.0,3.0\n4.0,5.0,6.0\n'
    >>> with open('example1.csv', 'w') as f:
    ...     f.write(s1)
    >>> a1 = read_from_filename('example1.csv')
    >>> a1
    array([[1., 2., 3.],
           [4., 5., 6.]])

    The second example has columns with different data types, so a
    one-dimensional array with a structured data type is returned.
    The tab character is used as the field delimiter.

    >>> s2 = '1.0\t10\talpha\n2.3\t25\tbeta\n4.5\t16\tgamma\n'
    >>> with open('example2.tsv', 'w') as f:
    ...     f.write(s2)
    >>> a2 = read_from_filename('example2.tsv', delimiter='\t')
    >>> a2
    array([(1. , 10, b'alpha'), (2.3, 25, b'beta'), (4.5, 16, b'gamma')],
          dtype=[('f0', '<f8'), ('f1', 'u1'), ('f2', 'S5')])
    """
    if usecols is not None:
        usecols = np.atleast_1d(np.array(usecols, dtype=np.int32))
        if usecols.ndim != 1:
            raise ValueError('usecols must be one-dimensional')

    try:
        operator.index(skiprows)
    except TypeError:
        raise TypeError("skiprows must be an integer") from None
    if skiprows < 0:
        raise ValueError("skiprows must be nonnegative")

    # Compute `codes` and `sizes`.  These are derived from `dtype`, and we
    # also pass `dtype` to the C function, so we're passing in redundant
    # information.  This is because it is easier to write the code that
    # creates `codes` and `sizes` using Python than C.
    if dtype is not None:
        codes, sizes = _flatten_dtype.flatten_dtype2(dtype)
        if (len(codes) > 1 and usecols is not None and
                len(codes) != len(usecols)):
            raise ValueError(f"length of usecols ({len(usecols)}) and "
                             f"number of fields in dtype ({len(codes)}) "
                             "do not match.")
    else:
        codes = None
        sizes = None

    if isinstance(file, str):
        arr = _readtext_from_filename(file, delimiter=delimiter, comment=comment,
                                      quote=quote, decimal=decimal, sci=sci,
                                      usecols=usecols, skiprows=skiprows,
                                      dtype=dtype, codes=codes, sizes=sizes)
    else:
        # Assume file is a file object.
        arr = _readtext_from_file_object(file, delimiter=delimiter, comment=comment,
                                         quote=quote, decimal=decimal, sci=sci,
                                         usecols=usecols, skiprows=skiprows,
                                         dtype=dtype, codes=codes, sizes=sizes)
    return arr
