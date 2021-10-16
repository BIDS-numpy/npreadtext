
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include "stream.h"
#include "tokenize.h"
#include "conversions.h"
#include "field_types.h"
#include "rows.h"
#include "growth.h"

/*
 * Minimum size to grow the allcoation by (or 25%). The 8KiB means the actual
 * growths is within `8 KiB <= size < 16 KiB` (depending on the row size).
 */
#define MIN_BLOCK_SIZE (1 << 13)


//
// If num_field_types is not 1, actual_num_fields must equal num_field_types.
//
static size_t
compute_row_size(
        int actual_num_fields, int num_field_types, field_type *field_types)
{
    size_t row_size;

    // rowsize is the number of bytes in each "row" of the array
    // filled in by this function.
    if (num_field_types == 1) {
        row_size = actual_num_fields * field_types[0].itemsize;
    }
    else {
        row_size = 0;
        for (int k = 0; k < num_field_types; ++k) {
            row_size += field_types[k].itemsize;
        }
    }
    return row_size;
}


/*
 *  Create the array of converter functions from the Python converters.
 */
PyObject **
create_conv_funcs(
        PyObject *converters, int num_fields, int32_t *usecols)
{
    PyObject **conv_funcs = PyMem_Calloc(num_fields, sizeof(PyObject *));
    if (conv_funcs == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    if (converters == Py_None) {
        return conv_funcs;
    }
    else if (!PyDict_Check(converters)) {
        PyErr_SetString(PyExc_TypeError,
                "converters must be a dictionary mapping columns to converter "
                "functions.");
        return NULL;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(converters, &pos, &key, &value)) {
        Py_ssize_t column = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (column == -1 && PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError,
                    "keys of the converters dictionary must be integers; "
                    "got %.100R", key);
            goto error;
        }
        if (usecols != NULL) {
            /*
             * This code searches for the corresponding usecol.  It is
             * identical to the legacy usecols code, which has two weaknesses:
             * 1. It fails for duplicated usecols only setting converter for
             *    the first one.
             * 2. It fails e.g. if usecols uses negative indexing and
             *    converters does not.  (This is a feature, since it allows
             *    us to correctly normalize converters to result column here.)
             */
            int i = 0;
            for (; i < num_fields; i++) {
                if (column == usecols[i]) {
                    column = i;
                    break;
                }
            }
            if (i == num_fields) {
                continue;  /* ignore unused converter */
            }
        }
        else {
            if (column < -num_fields || column >= num_fields) {
                PyErr_Format(PyExc_ValueError,
                        "converter specified for column %zd, which is invalid "
                        "for the number of fields %d.", column, num_fields);
                goto error;
            }
            if (column < 0) {
                column += num_fields;
            }
        }
        if (!PyCallable_Check(value)) {
            PyErr_Format(PyExc_TypeError,
                    "values of the converters dictionary must be callable, "
                    "but the value associated with key %R is not", key);
            goto error;
        }
        Py_INCREF(value);
        conv_funcs[column] = value;
    }
    return conv_funcs;

  error:
    for (int i = 0; i < num_fields; i++) {
        Py_XDECREF(conv_funcs[i]);
    }
    PyMem_FREE(conv_funcs);
    return NULL;
}

/**
 * Read a file into the provided array, or create (and possibly grow) an
 * array to read into.
 *
 * @param s The stream object/struct providing reading capabilities used by
 *        the tokenizer.
 * @param nrows Pointer to the number of rows to read, or -1.  If negative
 *        all rows are read.  The value is replaced with the actual number
 *        of rows read.
 * @param num_field_types The number of field types stored in `field_types`.
 * @param field_types Information about the dtype for each column (or one if
 *        `homogeneous`).
 * @param pconfig Pointer to the parser config object used by both the
 *        tokenizer and the conversion functions.
 * @param num_usecols The number of columns in `usecols`.
 * @param usecols An array of length `num_usecols` or NULL.  If given indicates
 *        which column is read for each individual row (negative columns are
 *        accepted).
 * @param skiplines The number of lines to skip, these lines are ignored.
 * @param converters Python dictionary of converters.  Finalizing converters
 *        is difficult without information about the number of columns.
 * @param data_array An array to be filled or NULL.  In either case a new
 *        reference is returned (the reference to `data_array` is not stolen).
 * @param out_descr The dtype used for allocating a new array.  This is not
 *        used if `data_array` is provided.  Note that the actual dtype of the
 *        returned array can differ for strings.
 * @param num_cols Pointer in which the actual (discovered) number of columns
 *        is returned.  This is only relevant if `homogeneous` is true.
 * @param homogeneous Whether the datatype of the array is not homogeneous,
 *        i.e. not structured.  In this case the number of columns has to be
 *        discovered an the returned array will be 2-dimensional rather than
 *        1-dimensional.
 *
 * @returns Returns the result as an array object or NULL on error.  The result
 *          is always a new reference (even when `data_array` was passed in).
 */
PyArrayObject *
read_rows(stream *s,
        Py_ssize_t *nrows, int num_field_types, field_type *field_types,
        parser_config *pconfig, int num_usecols, int *usecols,
        Py_ssize_t skiplines, PyObject *converters,
        PyArrayObject *data_array, PyArray_Descr *out_descr,
        bool homogeneous)
{
    char *data_ptr = NULL;
    int current_num_fields;
    size_t row_size;
    PyObject **conv_funcs = NULL;

    bool needs_init = PyDataType_FLAGCHK(out_descr, NPY_NEEDS_INIT);

    int ndim = homogeneous ? 2 : 1;
    npy_intp result_shape[2] = {0, 1};

    bool data_array_allocated = data_array == NULL;
    /* Make sure we own `data_array` for the purpose of error handling */
    Py_XINCREF(data_array);
    size_t rows_per_block = 1;  /* will be increased depending on row size */
    Py_ssize_t data_allocated_rows = 0;

    int ts_result = 0;
    tokenizer_state ts;
    if (tokenizer_init(&ts, pconfig) < 0) {
        goto error;
    }

    /* Set the actual number of fields if it is already known, otherwise -1 */
    int actual_num_fields = -1;
    if (usecols != NULL) {
        actual_num_fields = num_usecols;
    }
    else if (!homogeneous) {
        actual_num_fields = num_field_types;
    }

    for (; skiplines > 0; skiplines--) {
        ts.state = TOKENIZE_GOTO_LINE_END;
        ts_result = tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            goto error;
        }
        else if (ts_result != 0) {
            /* Fewer lines than skiplines is acceptable */
            break;
        }
    }

    Py_ssize_t row_count = 0;  /* number of rows actually processed */
    while ((*nrows < 0 || row_count < *nrows) && ts_result == 0) {
        ts_result = tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            goto error;
        }
        current_num_fields = ts.num_fields;
        field_info *fields = ts.fields;
        if (ts.num_fields == 0) {
            continue;  /* Ignore empty line */
        }

        if (NPY_UNLIKELY(data_ptr == NULL)) {
            // We've deferred some of the initialization tasks to here,
            // because we've now read the first line, and we definitively
            // know how many fields (i.e. columns) we will be processing.
            if (actual_num_fields == -1) {
                actual_num_fields = current_num_fields;
            }

            conv_funcs = create_conv_funcs(
                    converters, actual_num_fields, usecols);
            if (conv_funcs == NULL) {
                goto error;
            }

            /* Note that result_shape[1] is only used if homogeneous is true */
            result_shape[1] = actual_num_fields;
            /* row_size is used for growing the raw memory when necessary */
            row_size = compute_row_size(actual_num_fields,
                                        num_field_types, field_types);

            if (data_array == NULL) {
                if (*nrows < 0) {
                    /*
                     * Negative *nrows denotes to read the whole file, we
                     * approach this by allocating ever larger blocks.
                     * Adds a number of rows based on `MIN_BLOCK_SIZE`.
                     * Note: later code grows assuming this is a power of two.
                     */
                    if (row_size == 0) {
                        /* actual rows_per_block should not matter here */
                        rows_per_block = 512;
                    }
                    else {
                        /* safe on overflow since min_rows will be 0 or 1 */
                        size_t min_rows = (
                                (MIN_BLOCK_SIZE + row_size - 1) / row_size);
                        while (rows_per_block < min_rows) {
                            rows_per_block *= 2;
                        }
                    }
                    data_allocated_rows = rows_per_block;
                }
                else {
                    data_allocated_rows = *nrows;
                }
                result_shape[0] = data_allocated_rows;
                Py_INCREF(out_descr);
                /*
                 * We do not use Empty, as it would fill with None
                 * and requiring decref'ing if we shrink again.
                 */
                data_array = (PyArrayObject *)PyArray_SimpleNewFromDescr(
                        ndim, result_shape, out_descr);
                if (data_array == NULL) {
                    goto error;
                }
                if (needs_init) {
                    memset(PyArray_BYTES(data_array), 0, PyArray_NBYTES(data_array));
                }
            }
            else {
                assert(*nrows >=0);
                data_allocated_rows = *nrows;
            }
            data_ptr = PyArray_BYTES(data_array);
        }

        if (!usecols && (actual_num_fields != current_num_fields)) {
            PyErr_Format(PyExc_ValueError,
                    "the number of columns changed from %d to %d at row %zu; "
                    "use `usecols` to select a subset and avoid this error",
                    actual_num_fields, current_num_fields, row_count+1);
            goto error;
        }

        if (NPY_UNLIKELY(data_allocated_rows == row_count)) {
            /*
             * Grow by ~25% and rounded up to the next rows_per_block
             * NOTE: This is based on very crude timings and could be refined!
             */
            size_t new_rows = data_allocated_rows;
            npy_intp alloc_size = grow_size_and_multiply(
                    &new_rows, rows_per_block, row_size);
            if (alloc_size < 0) {
                /* should normally error much earlier, but make sure */
                PyErr_SetString(PyExc_ValueError,
                        "array is too big. Cannot read file as a single array; "
                        "providing a maximum number of rows to read may help.");
                goto error;
            }

            char *new_data = PyDataMem_RENEW(
                    PyArray_BYTES(data_array), alloc_size ? alloc_size : 1);
            if (new_data == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            /* Replace the arrays data since it may have changed */
            ((PyArrayObject_fields *)data_array)->data = new_data;
            ((PyArrayObject_fields *)data_array)->dimensions[0] = new_rows;
            data_ptr = new_data + row_count * row_size;
            data_allocated_rows = new_rows;
            if (needs_init) {
                memset(data_ptr, '\0', (new_rows - row_count) * row_size);
            }
        }

        for (int j = 0; j < actual_num_fields; ++j) {
            int f = homogeneous ? 0 : j;
            // k is the column index of the field in the file.
            int k;
            if (usecols == NULL) {
                k = j;
            }
            else {
                k = usecols[j];
                if (k < 0) {
                    // Python-like column indexing: k = -1 means the last column.
                    k += current_num_fields;
                }
                if (NPY_UNLIKELY((k < 0) || (k >= current_num_fields))) {
                    PyErr_Format(PyExc_ValueError,
                            "invalid column index %d at row %zu with %d "
                            "columns",
                            usecols[j], current_num_fields, row_count+1);
                    goto error;
                }
            }

            bool err = 0;
            Py_UCS4 *str = ts.field_buffer + fields[k].offset;
            Py_UCS4 *end = ts.field_buffer + fields[k + 1].offset - 1;
            if (conv_funcs[j] == NULL) {
                if (field_types[f].set_from_ucs4(field_types[f].descr,
                        str, end, data_ptr, pconfig) < 0) {
                    err = true;
                }
            }
            else {
                if (to_generic_with_converter(field_types[f].descr,
                        str, end, data_ptr, pconfig, conv_funcs[j]) < 0) {
                    err = true;
                }
            }
            data_ptr += field_types[f].itemsize;

            if (NPY_UNLIKELY(err)) {
                PyObject *exc, *val, *tb;
                PyErr_Fetch(&exc, &val, &tb);

                size_t length = end - str;
                PyObject *string = PyUnicode_FromKindAndData(
                        PyUnicode_4BYTE_KIND, str, length);
                if (string == NULL) {
                    npy_PyErr_ChainExceptions(exc, val, tb);
                    goto error;
                }
                PyErr_Format(PyExc_ValueError,
                        "could not convert string %.100R to %S at "
                        "row %zu, column %d.",
                        string, field_types[f].descr, row_count, k+1);
                Py_DECREF(string);
                npy_PyErr_ChainExceptionsCause(exc, val, tb);
                goto error;
            }
        }

        ++row_count;
    }

    tokenizer_clear(&ts);
    PyMem_FREE(conv_funcs);

    if (data_array == NULL) {
        assert(row_count == 0 && result_shape[0] == 0);
        if (actual_num_fields == -1) {
            /*
             * We found no rows and have to discover the number of elements
             * we have no choice but to guess 1.
             * NOTE: It may make sense to move this outside of here to refine
             *       the behaviour where necessary.
             */
            result_shape[1] = 1;
        }
        else {
            result_shape[1] = actual_num_fields;
        }
        Py_INCREF(out_descr);
        data_array = (PyArrayObject *)PyArray_Empty(
                ndim, result_shape, out_descr, 0);
    }

    /*
     * Note that if there is no data, `data_array` may still be NULL and
     * row_count is 0.  In that case, always realloc just in case.
     */
    if (data_array_allocated && data_allocated_rows != row_count) {
        size_t size = row_count * row_size;
        char *new_data = PyDataMem_RENEW(
                PyArray_BYTES(data_array), size ? size : 1);
        if (new_data == NULL) {
            Py_DECREF(data_array);
            PyErr_NoMemory();
            return NULL;
        }
        ((PyArrayObject_fields *)data_array)->data = new_data;
        ((PyArrayObject_fields *)data_array)->dimensions[0] = row_count;
    }

    *nrows = row_count;

    return data_array;

  error:
    PyMem_FREE(conv_funcs);
    tokenizer_clear(&ts);
    Py_XDECREF(data_array);
    return NULL;
}
