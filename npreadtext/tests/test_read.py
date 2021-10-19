import sys
from os import path
from io import StringIO
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, HAS_REFCOUNT
from npreadtext import read


def _get_full_name(basename):
    return path.join(path.dirname(__file__), 'data', basename)


@pytest.mark.parametrize('basename,delim', [('test1.csv', ','),
                                            ('test1.tsv', '\t')])
def test1_read1(basename, delim):
    filename = _get_full_name(basename)
    al = np.loadtxt(filename, delimiter=delim)

    a = read(filename, delimiter=delim)
    assert_array_equal(a, al)

    with open(filename, 'r') as f:
        a = read(f, delimiter=delim)
        assert_array_equal(a, al)


@pytest.mark.parametrize('basename,sci', [('test1e.csv', 'E'),
                                          ('test1d.csv', 'D')])
def test1_read_sci(basename, sci):
    filename = path.join(path.dirname(__file__), 'data', basename)
    filename_e = path.join(path.dirname(__file__), 'data', 'test1e.csv')
    al = np.loadtxt(filename_e, delimiter=',')

    a = read(filename, sci=sci)
    assert_array_equal(a, al)

    with open(filename, 'r') as f:
        a = read(f, sci=sci)
        assert_array_equal(a, al)


@pytest.mark.parametrize('basename,delim', [('test1.csv', ','),
                                            ('test1.tsv', '\t')])
def test1_read_usecols(basename, delim):
    filename = path.join(path.dirname(__file__), 'data', basename)
    al = np.loadtxt(filename, delimiter=delim, usecols=[0, 2])

    a = read(filename, usecols=[0, 2], delimiter=delim)
    assert_array_equal(a, al)

    with open(filename, 'r') as f:
        a = read(f, usecols=[0, 2], delimiter=delim)
        assert_array_equal(a, al)


def test1_read_with_comment():
    filename = _get_full_name('test1_with_comments.csv')
    al = np.loadtxt(filename, delimiter=',')

    a = read(filename, comment='#')
    assert_array_equal(a, al)

    with open(filename, 'r') as f:
        a = read(f, comment='#')
        assert_array_equal(a, al)


@pytest.mark.parametrize('comment', ['..', '//', '@-', 'this is the comment'])
def test_comment_multiple_chars(comment):
    content = '# IGNORE\n1.5,2.5# ABC\n3.0,4.0# XXX\n5.5,6.0\n'
    txt = StringIO(content.replace('#', comment))
    a = read(txt, dtype=np.float64, comment=comment, quote="")
    assert_equal(a, [[1.5, 2.5], [3.0, 4.0], [5.5, 6.0]])

def test_comment_multichar_error_with_quote():
    txt = StringIO("1,2\n3,4")
    with pytest.raises(ValueError):
        read(txt, comment="123")
    with pytest.raises(ValueError):
        read(txt, comment=["1", "3"])

    # a single character string in a tuple is unpacked though:
    a = read(txt, comment=("#",), quote='"')
    assert_equal(a, [[1, 2], [3, 4]])

def test_decimal_is_comma():
    filename = path.join(path.dirname(__file__),
                         'data', 'decimal_is_comma.txt')
    a = read(filename, decimal=',', delimiter=' ')
    expected = np.array([[1.5, 1.75, 2.0],
                         [2.5, 2.75, 3.0],
                         [3.5, 3.75, 4.0],
                         [4.5, 4.75, 5.0]])
    assert_array_equal(a, expected)


def test_quoted_field():
    filename = _get_full_name('quoted_field.csv')
    dtype = np.dtype([('f0', 'S8'), ('f1', np.float64)])

    a = read(filename, dtype=dtype)
    assert a.dtype == dtype
    expected = np.array([('alpha, x', 2.5),
                         ('beta, x', 4.5),
                         ('gamma, x', 5.0)], dtype=dtype)
    assert_array_equal(a, expected)


@pytest.mark.parametrize('skiprows', [0, 1, 3])
def test_dtype_and_skiprows(skiprows: int):
    filename = _get_full_name('mixed_types1.dat')

    dtype = np.dtype([('f0', np.uint16),
                      ('f1', np.float64),
                      ('f2', 'S7'),
                      ('f3', np.int8)])
    expected = np.array([(1000, 2.4, "alpha", -34),
                         (2000, 3.1, "beta", 29),
                         (3500, 9.9, "gamma", 120),
                         (4090, 8.1, "delta", 0),
                         (5001, 4.4, "epsilon", -99),
                         (6543, 7.8, "omega", -1)], dtype=dtype)

    a = read(filename, dtype=dtype, quote="'", delimiter=';', skiprows=skiprows)
    assert_array_equal(a, expected[skiprows:])


def test_structured_dtype_1():
    dt = [("a", 'u1', 2), ("b", 'u1', 2)]
    a = read(StringIO("0,1,2,3\n6,7,8,9\n"), dtype=dt)
    expected = np.array([((0, 1), (2, 3)), ((6, 7), (8, 9))],
                        dtype=dt)
    assert_array_equal(a, expected)


def test_structured_dtype_2():
    dt = [("a", 'u1', (2, 2))]
    a = read(StringIO("0 1 2 3"), delimiter=' ', dtype=dt)
    assert_array_equal(a, np.array([(((0, 1), (2, 3)),)], dtype=dt))


def test_nested_structured_subarray():
    # Test from numpygh-16678
    point = np.dtype([('x', float), ('y', float)])
    dt = np.dtype([('code', int), ('points', point, (2,))])
    res = read(StringIO('100,1,2,3,4\n200,5,6,7,8\n'), dtype=dt)

    expected = np.array([(100, [(1., 2.), (3., 4.)]),
                         (200, [(5., 6.), (7., 8.)])], dtype=dt)
    assert_array_equal(res, expected)


def test_structured_dtypes_offsets():
    # An aligned structured dtype will have additional padding:
    dt = np.dtype("i1,i4,i1,i4,i1,i4", align=True)
    res = read(StringIO('1,2,3,4,5,6\n7,8,9,10,11,12\n'), dtype=dt)

    expected = np.array([(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)], dtype=dt)
    assert_array_equal(res, expected)

@pytest.mark.parametrize('param', ['skiprows', 'max_rows'])
@pytest.mark.parametrize('badval, exc', [(-3, ValueError), (1.0, TypeError)])
def test_bad_nonneg_int(param, badval, exc):
    with pytest.raises(exc):
        read('foo.bar', **{param: badval})


@pytest.mark.parametrize('fn,shape', [('onerow.txt', (1, 5)),
                                      ('onecol.txt', (5, 1))])
def test_ndmin_single_row_or_col(fn, shape):
    filename = _get_full_name(fn)
    data = [1, 2, 3, 4, 5]
    arr2d = np.array(data).reshape(shape)

    a = read(filename, dtype=int, delimiter=' ')
    assert_array_equal(a, arr2d)

    a = read(filename, dtype=int, delimiter=' ', ndmin=0)
    assert_array_equal(a, data)

    a = read(filename, dtype=int, delimiter=' ', ndmin=1)
    assert_array_equal(a, data)

    a = read(filename, dtype=int, delimiter=' ', ndmin=2)
    assert_array_equal(a, arr2d)


@pytest.mark.parametrize('badval', [-1, 3, "plate of shrimp"])
def test_bad_ndmin(badval):
    with pytest.raises(ValueError):
        read('foo.bar', ndmin=badval)


def test_unpack_structured():
    filename = _get_full_name('mixed_types1.dat')
    dtype = np.dtype([('f0', np.uint16),
                      ('f1', np.float64),
                      ('f2', 'S7'),
                      ('f3', np.int8)])
    expected = np.array([(1000, 2.4, "alpha", -34),
                         (2000, 3.1, "beta", 29),
                         (3500, 9.9, "gamma", 120),
                         (4090, 8.1, "delta", 0),
                         (5001, 4.4, "epsilon", -99),
                         (6543, 7.8, "omega", -1)], dtype=dtype)
    a, b, c, d = read(
            filename, dtype=dtype, delimiter=';', quote="'", unpack=True)
    assert_array_equal(a, expected['f0'])
    assert_array_equal(b, expected['f1'])
    assert_array_equal(c, expected['f2'])
    assert_array_equal(d, expected['f3'])


def test_unpack_array():
    filename = _get_full_name('test1.csv')
    a, b, c = read(filename, delimiter=',', unpack=True)
    assert_array_equal(a, np.array([1.0, 4.0, 7.0, 0.0]))
    assert_array_equal(b, np.array([2.0, 5.0, 8.0, 1.0]))
    assert_array_equal(c, np.array([3.0, 6.0, 9.0, 2.0]))


@pytest.mark.parametrize("ws",
        "\t\u2003\u00A0\u3000") # tab and em, non-break, and ideographic space
def test_blank_lines_spaces_delimit(ws):
    # NOTE: It is unclear that the `  # comment` should succeed. Except
    #       for delimiter=None, which should use any whitespace (and maybe
    #       should just be implemented closer to Python).
    txt = StringIO(
        f'1 2{ws}30\n\n4 5 60\n  {ws}  \n7 8 {ws} 90\n  # comment\n3 2 1')
    a = read(txt, dtype=int, delimiter='', comment="#")
    assert_equal(a, np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]]))


def test_blank_lines_normal_delimiter():
    txt = StringIO('1,2,30\n\n4,5,60\n\n7,8,90\n# comment\n3,2,1')
    a = read(txt, dtype=int, delimiter=',', comment="#")
    assert_equal(a, np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]]))


def test_quoted_field_is_not_empty():
    txt = StringIO('1\n\n"4"\n""')
    a = read(txt, delimiter=",", dtype="U1")
    assert_equal(a, np.array([["1"], ["4"], [""]]))

@pytest.mark.parametrize("dtype", [np.float64, object])
def test_max_rows(dtype):
    txt = StringIO('1.5,2.5\n3.0,4.0\n5.5,6.0')
    a = read(txt, dtype=dtype, max_rows=2)
    assert_equal(a.dtype, dtype)
    assert_equal(a, np.array([["1.5", "2.5"], ["3.0", "4.0"]], dtype=dtype))


@pytest.mark.parametrize('dtype', [np.dtype('f8'), np.dtype('i2')])
def test_bad_values(dtype):
    txt = StringIO('1.5,2.5\n3.0,XXX\n5.5,6.0')
    msg = f"could not convert string .XXX. to {dtype} at row 1, column 2"
    with pytest.raises(ValueError, match=msg):
        read(txt, dtype=dtype)


def test_converters():
    txt = StringIO('1.5,2.5\n3.0,XXX\n5.5,6.0')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    a = read(txt, dtype=np.float64, converters=conv, encoding=None)
    assert_equal(a, [[1.5, 2.5], [3.0, np.nan], [5.5, 6.0]])


def test_converters_and_usecols():
    txt = StringIO('1.5,2.5,3.5\n3.0,4.0,XXX\n5.5,6.0,7.5\n')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    a = read(txt, dtype=np.float64, converters=conv,
             usecols=[0, -1], encoding=None)
    assert_equal(a, [[1.5, 3.5], [3.0, np.nan], [5.5, 7.5]])


def test_ragged_usecols():
    # Usecols, and negative ones, even work with variying number of columns
    txt = StringIO('0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n')
    a = read(txt, dtype=np.float64, usecols=[0, -2])
    assert_equal(a, [[0, 0], [0, 0], [0, 0]])


def test_empty_usecols():
    txt = StringIO('0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n')
    a = read(txt, dtype=np.dtype([]), usecols=[])
    assert a.shape == (3,)
    assert a.dtype == np.dtype([])


@pytest.mark.parametrize("c1", ["a", "の", "🫕"])
@pytest.mark.parametrize("c2", ["a", "の", "🫕"])
def test_large_unicode_characters(c1, c2):
    # c1 and c2 span ascii, 16bit and 32bit range.
    txt = StringIO(f"a,{c1},c,d\ne,{c2},f,g")
    a = read(txt, dtype=np.dtype('U12'))
    assert_equal(a, [f"a,{c1},c,d".split(","), f"e,{c2},f,g".split(",")])

def test_unicode_with_converter():
    txt = StringIO('cat,dog\nαβγ,δεζ\nabc,def\n')
    conv = {0: lambda s: s.upper()}
    a = read(txt, dtype=np.dtype('U12'), converters=conv, encoding=None)
    assert_equal(a, [['CAT', 'dog'], ['ΑΒΓ', 'δεζ'], ['ABC', 'def']])


def test_converter_with_structured_dtype():
    txt = StringIO('1.5,2.5,Abc\n3.0,4.0,dEf\n5.5,6.0,ghI\n')
    dt = np.dtype([('m', np.int32), ('r', np.float32), ('code', 'U8')])
    conv = {0: lambda s: int(10*float(s)), -1: lambda s: s.upper()}
    a = read(txt, dtype=dt, converters=conv)
    expected = np.array([(15, 2.5, 'ABC'), (30, 4.0, 'DEF'), (55, 6.0, 'GHI')],
                        dtype=dt)
    assert_equal(a, expected)


def test_read_huge_row():
    row = '1.5,2.5,' * 50000
    row = row[:-1] + "\n"
    txt = StringIO(row * 2)
    a = read(txt, delimiter=",", dtype=float)
    assert_equal(a, np.tile([1.5, 2.5], (2, 50000)))


def test_converter_with_unicode_dtype():
    """
    With the default 'bytes' encoding, tokens are encoded prior to converting.
    This means that the output of the converter may be bytes instead of
    unicode as expected by `read_rows`.
    This test checks that outputs from the above scenario are properly
    decoded before parsing by `read_rows`.
    """
    txt = StringIO('abc,def\nrst,xyz')
    conv = {col: bytes.upper for col in range(2)}
    a = read(txt, dtype=np.dtype('U3'), converters=conv)
    expected = np.array([['ABC', 'DEF'], ['RST', 'XYZ']])
    assert_equal(a, expected)


@pytest.mark.parametrize('dtype, actual_dtype', [('S', np.dtype('S5')),
                                                 ('U', np.dtype('U5'))])
def test_string_no_length_given(dtype, actual_dtype):
    # The given dtype is just 'S' or 'U', with no length.  In these
    # cases, the length of the resulting dtype is determined by the
    # longest string found in the file.
    txt = StringIO('AAA,5-1\nBBBBB,0-3\nC,4-9\n')
    a = read(txt, dtype=dtype)
    expected = np.array([['AAA', '5-1'], ['BBBBB', '0-3'], ['C', '4-9']],
                        dtype=actual_dtype)
    assert_equal(a.dtype, expected.dtype)
    assert_equal(a, expected)


def test_float_conversion():
    # Some tests that the conversion to float64 works as accurately
    # as the Python built-in `float` function.  In a naive version of
    # the float parser, these strings resulted in values that were off
    # by a ULP or two.
    strings = ['0.9999999999999999',
               '9876543210.123456',
               '5.43215432154321e+300',
               '0.901',
               '0.333']
    txt = StringIO('\n'.join(strings))
    a = read(txt)
    expected = np.array([float(s) for s in strings]).reshape((len(strings), 1))
    assert_equal(a, expected)


def test_bool():
    # Simple test for bool via integer:
    res = read(["1, 0", "10, -1"], dtype=bool)
    assert res.dtype == bool
    assert_array_equal(res, [[True, False], [True, True]])
    # Make sure we use only 1 and 0 on the byte-level:
    assert_array_equal(res.view(np.uint8), [[1, 0], [1, 1]])


@pytest.mark.parametrize('dt', [np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64])
def test_cast_float_to_int(dt):
    # Currently the parser_config flag 'allow_float_for_int' is hardcoded
    # to be true.  This means that if the parsing of an integer value
    # fails, the code will attempt to parse it as a float, and then
    # cast the float value to an integer.  This flag is only used when
    # an explicit dtype is given.
    txt = StringIO('1.0,2.1,3.7\n4,5,6')
    a = read(txt, dtype=dt)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=dt)
    assert_equal(a, expected)


@pytest.mark.parametrize('dt', [np.complex64, np.complex128])
@pytest.mark.parametrize('imaginary_unit', ['i', 'j'])
@pytest.mark.parametrize('with_parens', [False, True])
def test_complex(dt, imaginary_unit, with_parens):
    s = '(1.0-2.5j),3.75,(7+-5.0j)\n(4),(-19e2j),(0)'.replace('j',
                                                              imaginary_unit)
    if not with_parens:
        s = s.replace('(', '').replace(')', '')
    a = read(StringIO(s), dtype=dt, imaginary_unit=imaginary_unit)
    expected = np.array([[1.0-2.5j, 3.75, 7-5j], [4.0, -1900j, 0]], dtype=dt)
    assert_equal(a, expected)


def test_read_from_generator_1():

    def gen():
        for i in range(4):
            yield f'{i},{2*i},{i**2}'

    data = read(gen(), dtype=int)
    assert_equal(data, [[0, 0, 0], [1, 2, 1], [2, 4, 4], [3, 6, 9]])


def test_read_from_generator_2():

    def gen():
        for i in range(3):
            yield f'{i} {i/4}'

    data = read(gen(), dtype='i,d', delimiter=' ')
    expected = np.array([(0, 0.0), (1, 0.25), (2, 0.5)], dtype='i,d')
    assert_equal(data, expected)


def test_read_from_bad_generator():
    data = ["1,2", b"3,5", 12738]

    with pytest.raises(TypeError,
            match=r"non-string returned while reading data"):
        read(data, dtype='i,i', delimiter=',')


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_object_cleanup_on_read_error():
    sentinel = object()

    already_read = 0
    def conv(x):
        nonlocal already_read
        if already_read > 4999:
            raise ValueError("failed half-way through!")
        already_read += 1
        return sentinel

    txt = StringIO("x\n" * 10000)

    with pytest.raises(ValueError, match="at row 5000, column 1"):
        read(txt, dtype=object, converters={0: conv})

    assert sys.getrefcount(sentinel) == 2


def test_bad_encoding():
    data = [b"this is a byte string"]

    with pytest.raises(LookupError):
        read(data, encoding="this encoding is invalid!")


def test_character_not_bytes_compatible():
    # Test a character which cannot be encoded as "S".  Note that loadtxt
    # would try `string.encode("latin1")` throwing a slightly more informative
    # `UnicodeDecodeError`.
    data = ["–"]

    with pytest.raises(ValueError):
        read(data, dtype="S5")


def test_convert_raises_non_dict():
    data = StringIO("1 2\n3 4")
    with pytest.raises(TypeError, match="converters must be a dict"):
        read(data, converters=0)


def test_converter_raises_non_integer_key():
    data = StringIO("1 2\n3 4")
    with pytest.raises(TypeError, match="keys of the converters dict"):
        read(data, converters={"a": int})
    with pytest.raises(TypeError, match="keys of the converters dict"):
        read(data, converters={"a": int}, usecols=0)


@pytest.mark.parametrize("bad_col_ind", (3, -3))
def test_converter_raises_non_column_key(bad_col_ind):
    data = StringIO("1 2\n3 4")
    with pytest.raises(ValueError, match="converter specified for column"):
        read(data, converters={bad_col_ind: int})


def test_converter_raises_value_not_callable():
    data = StringIO("1 2\n3 4")
    with pytest.raises(
        TypeError, match="values of the converters dictionary must be callable"
    ):
        read(data, converters={0: 1})
