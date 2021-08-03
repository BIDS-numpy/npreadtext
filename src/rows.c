
#define _XOPEN_SOURCE 700

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <complex.h>

#include "stream.h"
#include "tokenize.h"
#include "sizes.h"
#include "conversions.h"
#include "field_types.h"
#include "rows.h"
#include "error_types.h"
#include "str_to.h"
#include "str_to_int.h"
#include "blocks.h"

#define INITIAL_BLOCKS_TABLE_LENGTH 200
#define ROWS_PER_BLOCK 500

#define ALLOW_PARENS true

//
// If num_field_types is not 1, actual_num_fields must equal num_field_types.
//
size_t compute_row_size(int actual_num_fields,
                        int num_field_types, field_type *field_types)
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


PyObject *call_converter_function(PyObject *func, char32_t *token, bool byte_converters)
{
    Py_ssize_t tokenlen = 0;
    while (token[tokenlen]) {
        ++tokenlen;
    }

    // Convert token to a Python unicode object
    PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, token, tokenlen);
    if (s == NULL) {
        // fprintf(stderr, "*** PyUnicode_FromKindAndData failed ***\n");
        return s;
    }

    // If byte_converters flag is set, encode prior to calling converter
    if (byte_converters) {
        s = PyObject_CallMethod(s, "encode", NULL);
        if (s == NULL) {
//            fprintf(stderr, "*** encode failed ***\n");
            return s;
        }
    }

    // Apply converter function
    PyObject *result = PyObject_CallFunctionObjArgs(func, s, NULL);
    Py_DECREF(s);
//    if (result == NULL) {
//        fprintf(stderr, "*** PyObject_CallFunctionObjArgs failed ***\n");
//    }
    return result;
}

/*
 *  Find the length of the longest token.
 */

size_t max_token_len(char32_t **tokens, int num_tokens,
                     int32_t *usecols, int num_usecols)
{
    size_t maxlen = 0;
    for (size_t i = 0; i < num_tokens; ++i) {
        size_t j;
        if (usecols == NULL) {
            j = i;
        }
        else {
            j = usecols[i];
        }
        size_t m = strlen32(tokens[j]);
        if (m > maxlen) {
            maxlen = m;
        }
    }
    return maxlen;
}


// WIP...
size_t max_token_len_with_converters(char32_t **tokens, int num_tokens,
                                     int32_t *usecols, int num_usecols,
                                     bool byte_converters, PyObject **conv_funcs)
{
    size_t maxlen = 0;
    size_t m;

    for (size_t i = 0; i < num_tokens; ++i) {
        size_t j;
        if (usecols == NULL) {
            j = i;
        }
        else {
            j = usecols[i];
        }

        if (conv_funcs && conv_funcs[j]) {
            PyObject *obj = call_converter_function(conv_funcs[i], tokens[j],
                                                    byte_converters);
            if (obj == NULL) {
//                fprintf(stderr, "CALL FAILED! strlen(tokens[j]) = %u, tokens[j] = %u\n",
//                                strlen32(tokens[j]), tokens[j][0]);
                return NULL;
            }
            // XXX check for obj == NULL!
            PyObject *s = PyObject_Str(obj);
            if (s == NULL) {
                // fprintf(stderr, "STR FAILED!\n");
                return NULL;
            }
            Py_DECREF(obj);
            // XXX check for s == NULL!
            Py_ssize_t len = PySequence_Length(s);
            if (len == -1) {
                // fprintf(stderr, "LEN FAILED!\n");
                return NULL;
            }
            // XXX check for len == -1
            Py_DECREF(s);
            m = (size_t) len;
        }
        else {
            m = strlen32(tokens[j]);
        }
        if (m > maxlen) {
            maxlen = m;
        }
    }
    return maxlen;
}


/*
 *  Create the array of converter functions from the Python converters dict.
 */
PyObject **create_conv_funcs(PyObject *converters, int32_t *usecols, int num_usecols,
                             int current_num_fields, read_error_type *read_error)
{
    PyObject **conv_funcs = NULL;
    size_t j, k;

    conv_funcs = calloc(num_usecols, sizeof(PyObject *));
    if (conv_funcs == NULL) {
        read_error->error_type = ERROR_OUT_OF_MEMORY;
        return NULL;
    }
    for (j = 0; j < num_usecols; ++j) {
        PyObject *key;
        PyObject *func;
        // k is the column index of the field in the file.
        if (usecols == NULL) {
            k = j;
        }
        else {
            k = usecols[j];
        }

        // XXX Check for failure of PyLong_FromSsize_t...
        key = PyLong_FromSsize_t((Py_ssize_t) k);
        func = PyDict_GetItem(converters, key);
        Py_DECREF(key);
        if (func == NULL) {
            key = PyLong_FromSsize_t((Py_ssize_t) k - current_num_fields);
            func = PyDict_GetItem(converters, key);
            Py_DECREF(key);
        }
        if (func != NULL) {
            Py_INCREF(func);
            conv_funcs[j] = func;
        }
    }
    return conv_funcs;
}

/*
 *  XXX Handle errors in any of the functions called by read_rows().
 *
 *  XXX Check handling of *nrows == 0.
 *
 *  Parameters
 *  ----------
 *  stream *s
 *  int *nrows
 *      On input, *nrows is the maximum number of rows to read.
 *      If *nrows is positive, `data_array` must point to the block of data
 *      where the data is to be stored.
 *      If *nrows is negative, all the rows in the file should be read, and
 *      the given value of `data_array` is ignored.  Data will be allocated
 *      dynamically in this function.
 *      On return, *nrows is the number of rows actually read from the file.
 *  int num_field_types
 *      Number of field types (i.e. the number of fields).  This is the
 *      length of the array pointed to by field_types.
 *  field_type *field_types
 *  parser_config *pconfig
 *  int32_t *usecols
 *      Pointer to array of column indices to use.
 *      If NULL, use all the columns (and ignore `num_usecols`).
 *  int num_usecols
 *      Length of the array `usecols`.  Ignored if `usecols` is NULL.
 *  int skiplines
 *  PyObject *converters
 *      dicitionary of converters
 *  void *data_array
 *  int *num_cols
 *      The actual number of columns (or fields) of the data being returned.
 *  read_error_type *read_error
 *      Information about errors detected in read_rows()
 */

void *read_rows(stream *s, int *nrows,
                int num_field_types, field_type *field_types,
                parser_config *pconfig,
                int32_t *usecols, int num_usecols,
                int skiplines,
                PyObject *converters,
                void *data_array,
                int *num_cols,
                read_error_type *read_error)
{
    char *data_ptr;
    int current_num_fields;
    char32_t **result;
    size_t row_size;
    size_t size;
    PyObject **conv_funcs = NULL;

    bool track_string_size = false;

    bool use_blocks;
    blocks_data *blks = NULL;

    int row_count;
    char32_t word_buffer[WORD_BUFFER_SIZE];
    int tok_error_type;

    int actual_num_fields = -1;

    read_error->error_type = 0;

    stream_skiplines(s, skiplines);

    if (stream_peek(s) == STREAM_EOF) {
        // There were fewer lines in the file than skiplines.
        // This is not treated as an error. The result should be an
        // empty array.

        //stream_close(s, RESTORE_FINAL);

        if (*nrows < 0) {
            *nrows = 0;
            return NULL;
        }
        else {
            *nrows = 0;
            return data_array;
        }
    }

    // track_string_size will be true if the user passes in
    // dtype=np.dtype('S') or dtype=np.dtype('U').  That is, the
    // data type is string or unicode, but a length was not given.
    // In this case, we must track the maximum length of the fields
    // and update the actual length for the dtype dynamically as
    // the file is read.
    track_string_size = ((num_field_types == 1) &&
                         (field_types[0].itemsize == 0) &&
                         ((field_types[0].typecode == 'S') ||
                          (field_types[0].typecode == 'U')));

    row_count = 0;
    while (((*nrows < 0) || (row_count < *nrows)) &&
           (result = tokenize(s, word_buffer, WORD_BUFFER_SIZE, pconfig,
                              &current_num_fields, &tok_error_type)) != NULL) {
        int j, k;

        if (actual_num_fields == -1) {
            // We've deferred some of the initialization tasks to here,
            // because we've now read the first line, and we definitively
            // know how many fields (i.e. columns) we will be processing.
            if (num_field_types > 1) {
                actual_num_fields = num_field_types;
            }
            else if (usecols != NULL) {
                actual_num_fields = num_usecols;
            }
            else {
                // num_field_types is 1.  (XXX Check that it can't be 0 or neg.)
                // Set actual_num_fields to the number of fields found in the
                // first line of data.
                actual_num_fields = current_num_fields;
            }

            if (usecols == NULL) {
                num_usecols = actual_num_fields;
            }
            else {
                // Normalize the values in usecols.
                for (j = 0; j < num_usecols; ++j) {
                    if (usecols[j] < 0) {
                        usecols[j] += current_num_fields;
                    }
                    // XXX Check that the value is between 0 and current_num_fields.
                }
            }

            if (converters != Py_None) {
                conv_funcs = create_conv_funcs(converters, usecols, num_usecols,
                                               current_num_fields, read_error);
                if (conv_funcs == NULL) {
                    return NULL;
                }
            }

            if (track_string_size) {
                // typecode must be 'S' or 'U'.
                // Find the maximum field length in the first line.
                size_t maxlen;
                if (converters != Py_None) {
                    maxlen = max_token_len_with_converters(result, actual_num_fields,
                                                           usecols, num_usecols,
                                                           pconfig->byte_converters,
                                                           conv_funcs);
                }
                else {
                    maxlen = max_token_len(result, actual_num_fields,
                                           usecols, num_usecols);
                }
                field_types[0].itemsize = (field_types[0].typecode == 'S') ? maxlen : 4*maxlen;
            }

            *num_cols = actual_num_fields;
            row_size = compute_row_size(actual_num_fields,
                                        num_field_types, field_types);

            use_blocks = false;
            if (*nrows < 0) {
                // Any negative value means "read the entire file".
                // In this case, it is assumed that *data_array is NULL
                // or not initialized. I.e. the value passed in is ignored,
                // and instead is initialized to the first block.
                use_blocks = true;
                blks = blocks_init(row_size, ROWS_PER_BLOCK, INITIAL_BLOCKS_TABLE_LENGTH);
                if (blks == NULL) {
                    // XXX Check for other clean up that might be necessary.
                    read_error->error_type = ERROR_OUT_OF_MEMORY;
                    free(conv_funcs);
                    return NULL;
                }
            }
            else {
                // *nrows >= 0
                // FIXME: Ensure that *nrows == 0 is handled correctly.
                if (data_array == NULL) {
                    // The number of rows to read was given, but a memory buffer
                    // was not, so allocate one here.
                    size = *nrows * row_size;
                    data_array = malloc(size);
                    if (data_array == NULL) {
                        read_error->error_type = ERROR_OUT_OF_MEMORY;
                        return NULL;
                    }
                }
                data_ptr = data_array;
            }
        }
        else {
            // *Not* the first line...
            if (track_string_size) {
                size_t new_itemsize, maxlen;
                // typecode must be 'S' or 'U'.
                // Find the maximum field length in the current line.
                if (converters != Py_None) {
                    maxlen = max_token_len_with_converters(result, actual_num_fields,
                                                           usecols, num_usecols,
                                                           pconfig->byte_converters,
                                                           conv_funcs);
                }
                else {
                    maxlen = max_token_len(result, actual_num_fields,
                                                  usecols, num_usecols);
                }
                new_itemsize = (field_types[0].typecode == 'S') ? maxlen : 4*maxlen;
                if (new_itemsize > field_types[0].itemsize) {
                    // There is a field in this row whose length is
                    // more than any previously seen length.
                    if (use_blocks) {
                        int status = blocks_uniform_resize(blks, actual_num_fields, new_itemsize);
                        if (status != 0) {
                            // XXX Handle this--probably out of memory.
                        }
                    }
                    field_types[0].itemsize = new_itemsize;
                }
            }    
        }

        if (!usecols && (actual_num_fields != current_num_fields)) {
            read_error->error_type = ERROR_CHANGED_NUMBER_OF_FIELDS;
            read_error->line_number = stream_linenumber(s);
            read_error->column_index = current_num_fields;
            if (use_blocks) {
                blocks_destroy(blks);
            }
            return NULL;
        }

        if (use_blocks) {
            data_ptr = blocks_get_row_ptr(blks, row_count);
            if (data_ptr == NULL) {
                blocks_destroy(blks);
                read_error->error_type = ERROR_OUT_OF_MEMORY;
                return NULL;
            }
        }

        for (j = 0; j < num_usecols; ++j) {
            int error = ERROR_OK;
            // f is the index into the field_types array.  If there is only
            // one field type, it applies to all fields found in the file.
            int f = (num_field_types == 1) ? 0 : j;
            char typecode = field_types[f].typecode;
            PyObject *converted = NULL;

            // k is the column index of the field in the file.
            if (usecols == NULL) {
                k = j;
            }
            else {
                k = usecols[j];
                if (k < 0) {
                    // Python-like column indexing: k = -1 means the last column.
                    k += current_num_fields;
                }
                if ((k < 0) || (k >= current_num_fields)) {
                    read_error->error_type = ERROR_INVALID_COLUMN_INDEX;
                    read_error->line_number = stream_linenumber(s) - 1;
                    read_error->column_index = usecols[j];
                    break;
                }
            }

            read_error->error_type = 0;
            read_error->line_number = stream_linenumber(s) - 1;
            read_error->field_number = k;
            read_error->char_position = -1; // FIXME
            read_error->typecode = typecode;

            if (conv_funcs && conv_funcs[j]) {
                converted = call_converter_function(conv_funcs[j], result[k],
                                                    pconfig->byte_converters);
                if (converted == NULL) {
                    read_error->error_type = ERROR_CONVERTER_FAILED;
                    break;
                }
            }

            // FIXME: There is a lot of repeated code in the following.
            // This needs to be modularized.

            if (typecode == 'b') {
                int8_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        long long value = PyLong_AsLongLong(converted);
                        if (value == -1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (int8_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_int8(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(int8_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'B') {
                uint8_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        size_t value = PyLong_AsSize_t(converted);
                        if (value == (size_t)-1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (uint8_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_uint8(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(uint8_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'h') {
                int16_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        long long value = PyLong_AsLongLong(converted);
                        if (value == -1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (int16_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_int16(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(int16_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'H') {
                uint16_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        size_t value = PyLong_AsSize_t(converted);
                        if (value == (size_t)-1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (uint16_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_uint16(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(uint16_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'i') {
                int32_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        long long value = PyLong_AsLongLong(converted);
                        if (value == -1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (int32_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_int32(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(int32_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'I') {
                uint32_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        size_t value = PyLong_AsSize_t(converted);
                        if (value == (size_t)-1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (uint32_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_uint32(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(uint32_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'q') {
                int64_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        long long value = PyLong_AsLongLong(converted);
                        if (value == -1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (int64_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_int64(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(int64_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'Q') {
                uint64_t x = 0;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        size_t value = PyLong_AsSize_t(converted);
                        if (value == (size_t)-1) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                        x = (uint64_t) value;  // FIXME: Check out of bounds!
                    }
                    else {
                        x = to_uint64(result[k], pconfig, &read_error->error_type);
                        if (read_error->error_type) {
                            break;
                        }
                    }
                }
                *(uint64_t *) data_ptr = x;
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'f' || typecode == 'd') {
                // Convert to float.
                double x = NAN;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        x = PyFloat_AsDouble(converted);
                        if (x == -1.0) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                    }
                    else {
                        char32_t decimal = pconfig->decimal;
                        char32_t sci = pconfig->sci;
                        if ((*(result[k]) == '\0') || !to_double(result[k], &x, sci, decimal)) {
                            read_error->error_type = ERROR_BAD_FIELD;
                            break;
                        }
                    }
                }
                if (typecode == 'f') {
                    *(float *) data_ptr = (float) x;
                }
                else {
                    *(double *) data_ptr = x;
                }
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'z' || typecode == 'c') {
                // Convert to complex.
                double x = NAN;
                double y = NAN;
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        // FIXME: This case not converted from the float/double code.
                        x = PyFloat_AsDouble(converted);
                        if (x == -1.0) {
                            if (PyErr_Occurred()) {
                                read_error->error_type = ERROR_BAD_FIELD;
                                break;
                            }
                        }
                    }
                    else {
                        char32_t decimal = pconfig->decimal;
                        char32_t sci = pconfig->sci;
                        char32_t imaginary_unit = pconfig->imaginary_unit;
                        if ((*(result[k]) == '\0') || !to_complex(result[k], &x, &y,
                                                                  sci, decimal,
                                                                  imaginary_unit,
                                                                  ALLOW_PARENS)) {
                            read_error->error_type = ERROR_BAD_FIELD;
                            break;
                        }
                    }
                }
                if (typecode == 'c') {
                    *(complex float *) data_ptr = (complex float) (x + I*y);
                }
                else {
                    *(complex double *) data_ptr = x + I*y;
                }
                data_ptr += field_types[f].itemsize;
            }
            else if (typecode == 'S') {
                // String
                memset(data_ptr, 0, field_types[f].itemsize);
                if (k < current_num_fields) {
                    //strncpy(data_ptr, result[k], field_types[f].itemsize);
                    size_t i = 0;
                    while (i < (size_t) field_types[f].itemsize && result[k][i]) {
                        data_ptr[i] = result[k][i];
                        ++i;
                    }
                    if (i < (size_t) field_types[f].itemsize) {
                        data_ptr[i] = '\0';
                    }
                }
                data_ptr += field_types[f].itemsize;
            }
            else {  // typecode == 'U'
                memset(data_ptr, 0, field_types[f].itemsize);
                if (k < current_num_fields) {
                    if (converted != NULL) {
                        Py_ssize_t len;
                        int kind;
                        void *data;
                        // The value returned by the converter may be a bytes
                        // object when e.g. `encoding="bytes"`. In this case,
                        // decode before processing
                        if (PyBytes_Check(converted)) {
                            PyObject* converted_unicode = PyObject_CallMethod(
                                converted, "decode", NULL);
                            converted = converted_unicode;
                            Py_DECREF(converted_unicode);
                        }
                        if (!PyUnicode_Check(converted)) {
                            read_error->error_type = ERROR_BAD_FIELD;
                            break;
                        }
                        len = PyUnicode_GET_LENGTH(converted);
                        if (4*len > field_types[f].itemsize) {
                            // XXX Make a more specific error type? Converted
                            // Unicode string is too long.
                            read_error->error_type = ERROR_BAD_FIELD;
                            break;
                        }
                        kind = PyUnicode_KIND(converted);
                        data = PyUnicode_DATA(converted);
                        for (Py_ssize_t i = 0; i < len; ++i) {
                            *(char32_t *)(data_ptr + 4*i) = PyUnicode_READ(kind, data, i);
                        }
                    }
                    else {
                        size_t i = 0;
                        // XXX The '4's in the following are sizeof(char32_t).
                        while (i < (size_t) field_types[f].itemsize/4 && result[k][i]) {
                            *(char32_t *)(data_ptr + 4*i) = result[k][i];
                            ++i;
                        }
                        if (i < (size_t) field_types[f].itemsize/4) {
                            *(char32_t *)(data_ptr + 4*i) = '\0';
                        }
                    }
                }
                data_ptr += field_types[f].itemsize;
            }
        }

        free(result);

        if (read_error->error_type != 0) {
            break;
        }

        ++row_count;
    }

    if (use_blocks) {
        if (read_error->error_type == 0) {
            // No error.
            // Copy the blocks into a newly allocated contiguous array.
            data_array = blocks_to_contiguous(blks, row_count);
        }
        blocks_destroy(blks);
    }

    //stream_close(s, RESTORE_FINAL);

    *nrows = row_count;

    if (read_error->error_type) {
        return NULL;
    }
    return (void *) data_array;
}
