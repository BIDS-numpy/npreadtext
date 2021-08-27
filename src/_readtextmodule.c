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
#include "stream_file.h"
#include "stream_python_file_by_line.h"
#include "field_types.h"
#include "analyze.h"
#include "rows.h"
#include "error_types.h"

#define LOADTXT_COMPATIBILITY true


static void
raise_analyze_exception(int nrows, char *filename)
{
    if (nrows == ANALYZE_OUT_OF_MEMORY) {
        if (filename) {
            PyErr_Format(PyExc_MemoryError,
                         "Out of memory while analyzing '%s'", filename);
        } else {
            PyErr_Format(PyExc_MemoryError,
                         "Out of memory while analyzing file.");
        }
    }
    else if (nrows == ANALYZE_FILE_ERROR) {
        if (filename) {
            PyErr_Format(PyExc_RuntimeError,
                         "File error while analyzing '%s'", filename);
        } else {
            PyErr_Format(PyExc_RuntimeError,
                         "File error while analyzing file.");
        }
    }
    else {
        if (filename) {
            PyErr_Format(PyExc_RuntimeError,
                         "Unknown error when analyzing '%s'", filename);
        } else {
            PyErr_Format(PyExc_RuntimeError,
                         "Unknown error when analyzing file.");
        }
    }
}


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
_readtext_from_stream(stream *s, char *filename, parser_config *pc,
                      PyObject *usecols, Py_ssize_t skiprows, Py_ssize_t max_rows,
                      PyObject *converters,
                      PyObject *dtype, PyArray_Descr **dtypes,
                      int num_dtype_fields)
{
    PyObject *arr = NULL;
    PyArray_Descr *out_dtype = NULL;
    int32_t *cols;
    int ncols;
    npy_intp nrows;
    int num_fields;
    field_type *ft = NULL;

    bool homogeneous;
    bool needs_init = false;
    npy_intp shape[2];

    if (dtype == Py_None) {
        // Make the first pass of the file to analyze the data type
        // and count the number of rows.
        // analyze() will assign num_fields and create the ft array,
        // based on the types of the data that it finds in the file.
        // XXX Note that analyze() does not use the usecols data--it
        // analyzes (and fills in ft for) all the columns in the file.
        nrows = analyze(s, pc, skiprows, -1, &num_fields, &ft);
        if (nrows < 0) {
            raise_analyze_exception(nrows, filename);
            return NULL;
        }
        stream_seek(s, 0);
        if (nrows == 0) {
            // Empty file, and a dtype was not given.  In this case, return
            // an array with shape (0, 0) and data type float64.
            npy_intp dims[2] = {0, 0};
            arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
            goto finish;
        }
        homogeneous = field_types_is_homogeneous(num_fields, ft);
        if (field_types_init_descriptors(num_fields, ft) < 0) {
            goto finish;
        }
        if (homogeneous) {
            out_dtype = ft[0].descr;
            Py_INCREF(out_dtype);
        }
        else {
            out_dtype = field_types_to_descr(num_fields, ft);
            if (out_dtype == NULL) {
                goto finish;
            }
        }
    }
    else {
        /*
         * If dtypes[0] is dtype the input was not structured and the result
         * is considered "homogeneous" and we have to discover the number of
         * columns/
         */
        out_dtype = (PyArray_Descr *)dtype;
        Py_INCREF(out_dtype);
        needs_init = PyDataType_FLAGCHK(out_dtype, NPY_NEEDS_INIT);

        /* TODO: Ridiculous, should just pass it in (or reuse num_fields) */
        homogeneous = num_dtype_fields == 1 && (out_dtype == dtypes[0]);

        // A dtype was given.
        num_fields = num_dtype_fields;
        ft = field_types_create(num_fields, dtypes);
        if (ft == NULL) {
            PyErr_Format(PyExc_MemoryError, "out of memory");
            return NULL;
        }
        nrows = max_rows;
    }
    if (usecols == Py_None) {
        ncols = num_fields;
        cols = NULL;
    }
    else {
        ncols = PyArray_SIZE(usecols);
        cols = PyArray_DATA(usecols);
    }

    // XXX In the one-pass case, we don't have nrows.
    shape[0] = nrows;
    if (homogeneous) {
        shape[1] = ncols;
    }

    if (dtype == Py_None) {
        int num_cols;
        int ndim = homogeneous ? 2 : 1;

        Py_INCREF(out_dtype);
        arr = PyArray_SimpleNewFromDescr(ndim, shape, out_dtype);
        if (!arr) {
            goto finish;
        }
        Py_ssize_t num_rows = nrows;
        void *result = read_rows(s,
                &num_rows, num_fields, ft, pc, cols, ncols, skiprows,
                converters, PyArray_DATA(arr), &num_cols, homogeneous,
                needs_init /* unused, data is allocated and initialized */);
        if (result == NULL) {
            goto finish;
        }
    }
    else {
        // A dtype was given.
        int num_cols;
        int ndim;
        Py_ssize_t num_rows = nrows;
        bool track_string_size = ((num_dtype_fields == 1) &&
                                  (ft[0].itemsize == 0) &&
                                  ((ft[0].typecode == 'S') ||
                                   (ft[0].typecode == 'U')));
        if (track_string_size) {
            /*
             * The string size tracking currently mutates the descriptor,
             * so we have to make a copy that we own (but can use it later)
             */
            ft[0].descr = PyArray_DescrNewFromType(ft[0].descr->type_num);
            if (ft[0].descr == NULL) {
                goto finish;
            }
        }
        void *result = read_rows(s, &num_rows, num_fields, ft, pc,
                                 cols, ncols, skiprows, converters,
                                 NULL, &num_cols, homogeneous, needs_init);
        if (result == NULL) {
            goto finish;
        }

        shape[0] = num_rows;
        if (PyDataType_ISSTRING(out_dtype) || !PyDataType_ISEXTENDED(out_dtype)) {
            ndim = 2;
            if (num_rows > 0) {
                shape[1] = num_cols;
            }
            else {
                // num_rows == 0 => empty file.
                shape[1] = 0;
            }
        }
        else {
            ndim = 1;
            shape[1] = 1;  // Not actually necessary to fill this in.
        }
        if (track_string_size) {
            /* The reading modified `ft[0]` in-place, use it */
            PyArray_Descr *dt = ft[0].descr;
            Py_INCREF(dt);
            // XXX Fix memory management - `result` was malloc'd.
            arr = PyArray_NewFromDescr(&PyArray_Type, dt,
                                       ndim, shape, NULL, result, 0, NULL);
        }
        else {
            // We have to INCREF dtype, because the Python caller owns a
            // reference, and PyArray_NewFromDescr steals a reference to it.
            Py_INCREF(out_dtype);
            // XXX Fix memory management - `result` was malloc'd.
            arr = PyArray_NewFromDescr(&PyArray_Type, out_dtype,
                                       ndim, shape, NULL, result, 0, NULL);
        }
        if (!arr) {
            free(ft);
            free(result);
            return NULL;
        }
    }

  finish:
    Py_XDECREF(out_dtype);
    if (ft != NULL) {
        field_types_clear(num_fields, ft);
        free(ft);
    }
    return arr;
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
        /* TODO: This sets it to a non-existing character, could use NUL */
        *character = (Py_UCS4)-1;
        return 1;
    }
    *character = PyUnicode_READ_CHAR(obj, 0);
    return 1;
}


static PyObject *
_readtext_from_file_object(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"file", "delimiter", "comment", "quote",
                             "decimal", "sci", "imaginary_unit",
                             "usecols", "skiprows",
                             "max_rows", "converters",
                             "dtype", "dtypes",
                             "encoding", NULL};
    PyObject *file;
    Py_ssize_t skiprows = 0;
    Py_ssize_t max_rows = -1;
    PyObject *usecols = Py_None;
    PyObject *converters = Py_None;

    PyObject *dtype = Py_None;
    PyObject *dtypes_obj = Py_None;
    PyObject *encoding = Py_None;

    PyArray_Descr **dtypes = NULL;

    parser_config pc;
    pc.delimiter = ',';
    pc.comment = '#';
    pc.quote = '"';
    pc.decimal = '.';
    pc.sci = 'E';
    pc.imaginary_unit = 'j';
    pc.allow_float_for_int = true;
    pc.allow_embedded_newline = true;
    pc.delimiter_is_whitespace = false;
    pc.ignore_leading_whitespace = false;
    pc.ignore_trailing_spaces = false;
    pc.ignore_blank_lines = true;
    pc.strict_num_fields = false;

    PyObject *arr = NULL;
    int num_dtype_fields;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|$O&O&O&O&O&O&OnnOOOO", kwlist,
            &file,
            &parse_control_character, &pc.delimiter,
            &parse_control_character, &pc.comment,
            &parse_control_character, &pc.quote,
            &parse_control_character, &pc.decimal,
            &parse_control_character, &pc.sci,
            &parse_control_character, &pc.imaginary_unit,
            &usecols, &skiprows, &max_rows, &converters,
            &dtype, &dtypes_obj, &encoding)) {
        return NULL;
    }

    if (pc.delimiter == (Py_UCS4)-1) {
        /* TODO: We can allow a '\0' delimiter; need to refine argparsing */
        pc.delimiter_is_whitespace = true;
        /* Ignore leading whitespace to match `string.split(None)` */
        pc.ignore_leading_whitespace = true;
    }

    /*
     * TODO: This needs some hefty input validation!
     */
    if (dtypes_obj == Py_None) {
        assert(dtype == Py_None);
        num_dtype_fields = -1;
    }
    else {
        dtypes_obj = PySequence_Fast(dtypes_obj, "dtypes not a sequence :(");
        if (dtypes_obj == NULL) {
            return NULL;
        }
        num_dtype_fields = PySequence_Fast_GET_SIZE(dtypes_obj);
        dtypes = (PyArray_Descr **)PySequence_Fast_ITEMS(dtypes_obj);
    }

    stream *s = stream_python_file_by_line(file, encoding);
    if (s == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Unable to access the file.");
        Py_DECREF(dtypes_obj);
        return NULL;
    }

    arr = _readtext_from_stream(s, NULL, &pc, usecols, skiprows, max_rows,
                                converters,
                                dtype, dtypes, num_dtype_fields);
    Py_DECREF(dtypes_obj);
    stream_close(s, RESTORE_NOT);
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

    //
    // Initialize numpy.
    //
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }

    // ----------------------------------------------------------------
    // Finish the extension module creation.
    // ----------------------------------------------------------------  

    // Create module
    m = PyModule_Create(&moduledef);

    return m;
}
