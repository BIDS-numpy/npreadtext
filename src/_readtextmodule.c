//
//  Requires C99.
//

#include <stdio.h>
#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/arrayobject.h"

#include "parser_config.h"
#include "stream_pyobject.h"
#include "field_types.h"
#include "rows.h"
#include "str_to_int.h"


//
// `usecols` must point to a Python object that is Py_None or a 1-d contiguous
// numpy array with data type int32.
//
// `dtype` must point to a Python object that is Py_None or a numpy dtype
// instance.  If the latter, code and sizes must be arrays of length
// num_dtype_fields, holding the flattened data field type codes and byte
// sizes. (num_dtype_fields, codes, and sizes can be inferred from dtype,
// but we do that in Python code.)
//
// If both `usecols` and `dtype` are not None, and the data type is compound,
// then len(usecols) must equal num_dtype_fields.
//
// If `dtype` is given and it is compound, and `usecols` is None, then the
// number of columns in the file must match the number of fields in `dtype`.
//
static PyObject *
_readtext_from_stream(stream *s, parser_config *pc,
                      PyObject *usecols, Py_ssize_t skiprows, Py_ssize_t max_rows,
                      PyObject *converters, PyObject *dtype)
{
    PyArrayObject *arr = NULL;
    PyArray_Descr *out_dtype = NULL;
    int32_t *cols;
    int ncols;
    field_type *ft = NULL;

    /*
     * If dtypes[0] is dtype the input was not structured and the result
     * is considered "homogeneous" and we have to discover the number of
     * columns/
     */
    out_dtype = (PyArray_Descr *)dtype;
    Py_INCREF(out_dtype);

    npy_intp num_fields = field_types_create(out_dtype, &ft);
    if (num_fields < 0) {
        goto finish;
    }
    bool homogeneous = num_fields == 1 && ft[0].descr == out_dtype;

    if (usecols == Py_None) {
        ncols = num_fields;
        cols = NULL;
    }
    else {
        ncols = PyArray_SIZE(usecols);
        cols = PyArray_DATA(usecols);
    }

    arr = read_rows(
            s, max_rows, num_fields, ft, pc,
            ncols, cols, skiprows, converters,
            NULL, out_dtype, homogeneous);
    if (arr == NULL) {
        goto finish;
    }

  finish:
    Py_XDECREF(out_dtype);
    field_types_xclear(num_fields, ft);
    return (PyObject *)arr;
}


static int
parse_control_character(PyObject *obj, Py_UCS4 *character)
{
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) > 1) {
        PyErr_Format(PyExc_TypeError,
                "Control character must be a single unicode character or "
                "empty unicode string; but got: %.100R", obj);
        return 0;
    }
    if (PyUnicode_GET_LENGTH(obj) == 0) {
        *character = (Py_UCS4)-1;  /* character beyond unicode range */
        return 1;
    }
    *character = PyUnicode_READ_CHAR(obj, 0);
    return 1;
}


static PyObject *
_readtext_from_file_object(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"file", "delimiter", "comment", "quote",
                             "imaginary_unit", "usecols", "skiprows",
                             "max_rows", "converters", "dtype",
                             "encoding", "filelike",
                             "byte_converters", "c_byte_converters",
                             NULL};
    PyObject *file;
    Py_ssize_t skiprows = 0;
    Py_ssize_t max_rows = -1;
    PyObject *usecols = Py_None;
    PyObject *converters = Py_None;

    PyObject *dtype = Py_None;
    char *encoding = NULL;
    int filelike = 1;

    parser_config pc = {
        .delimiter = ',',
        .comment = '#',
        .quote = '"',
        .imaginary_unit = 'j',
        .allow_float_for_int = true,
        .allow_embedded_newline = true,
        .delimiter_is_whitespace = false,
        .ignore_leading_whitespace = false,
        .python_byte_converters = false,
        .c_byte_converters = false,
    };
    int python_byte_converters = 0;
    int c_byte_converters = 0;

    PyObject *arr = NULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|$O&O&O&O&OnnOOzppp", kwlist,
            &file,
            &parse_control_character, &pc.delimiter,
            &parse_control_character, &pc.comment,
            &parse_control_character, &pc.quote,
            &parse_control_character, &pc.imaginary_unit,
            &usecols, &skiprows, &max_rows, &converters,
            &dtype, &encoding, &filelike,
            &python_byte_converters, &c_byte_converters)) {
        return NULL;
    }
    pc.python_byte_converters = python_byte_converters;
    pc.c_byte_converters = c_byte_converters;

    if (pc.delimiter == (Py_UCS4)-1) {
        pc.delimiter_is_whitespace = true;
        /* Ignore leading whitespace to match `string.split(None)` */
        pc.ignore_leading_whitespace = true;
    }

    if (!PyArray_DescrCheck(dtype) ) {
        PyErr_SetString(PyExc_TypeError,
                "internal error: dtype must be provided and be a NumPy dtype");
        return NULL;
    }

    stream *s;
    if (filelike) {
        s = stream_python_file(file, encoding);
    }
    else {
        s = stream_python_iterable(file, encoding);
    }
    if (s == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Unable to access the file.");
        return NULL;
    }

    arr = _readtext_from_stream(s, &pc, usecols, skiprows, max_rows,
                                converters, dtype);
    stream_close(s);
    return arr;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definition.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

PyMethodDef module_methods[] = {
    {"_readtext_from_file_object", (PyCFunction) _readtext_from_file_object,
         METH_VARARGS | METH_KEYWORDS, "testing"},
    {0} // sentinel
};

static struct PyModuleDef moduledef = {
    .m_base     = PyModuleDef_HEAD_INIT,
    .m_name     = "_readtextmodule",
    .m_size     = -1,
    .m_methods  = module_methods,
};


PyMODINIT_FUNC
PyInit__readtextmodule(void)
{
    PyObject* m = NULL;

    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    double_descr = PyArray_DescrFromType(NPY_DOUBLE);

    m = PyModule_Create(&moduledef);

    return m;
}
