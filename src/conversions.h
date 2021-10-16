#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <stdbool.h>

#include "parser_config.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include "numpy/arrayobject.h"

int
to_bool(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

int
to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

int
to_double(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

int
to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

int
to_cdouble(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

int
to_string(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused);

int
to_unicode(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused);

int
to_generic_with_converter(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused, PyObject *func);

int
to_generic(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

#endif
