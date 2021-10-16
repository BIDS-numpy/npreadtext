#include "field_types.h"
#include "conversions.h"
#include "str_to_int.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include <numpy/arrayobject.h>


void
field_types_xclear(int num_field_types, field_type *ft) {
    if (ft == NULL) {
        return;
    }
    for (int i = 0; i < num_field_types; i++) {
        Py_XDECREF(ft[i].descr);
        ft[i].descr = NULL;
    }
    PyMem_Free(ft);
}


/*
 * Prepare the "field_types" for the given dtypes/descriptors.  Currently,
 * we copy the itemsize, but the main thing is that we check for custom
 * converters.
 * TODO: As soon as we move this into NumPy we could move the conversion
 *       function onto the DType.  At this point, this would not be used or
 *       only used as a fast storage.
 */
field_type *
field_types_create(int num_field_types, PyArray_Descr *dtypes[])
{
    field_type *ft;

    ft = PyMem_Malloc(num_field_types * sizeof(field_type));
    if (ft == NULL) {
        return NULL;
    }
    for (int i = 0; i < num_field_types; ++i) {
        PyArray_Descr *descr = dtypes[i];
        Py_INCREF(descr);
        ft[i].descr = descr;

        if (descr->type_num == NPY_BOOL) {
            ft[i].set_from_ucs4 = &to_bool;
        }
        else if (PyDataType_ISSIGNED(descr)) {
            switch (descr->elsize) {
                case 1:
                    ft[i].set_from_ucs4 = &to_int8;
                    break;
                case 2:
                    ft[i].set_from_ucs4 = &to_int16;
                    break;
                case 4:
                    ft[i].set_from_ucs4 = &to_int32;
                    break;
                case 8:
                    ft[i].set_from_ucs4 = &to_int64;
                    break;
                default:
                    assert(0);
            }
        }
        else if (PyDataType_ISUNSIGNED(descr)) {
            switch (descr->elsize) {
                case 1:
                    ft[i].set_from_ucs4 = &to_uint8;
                    break;
                case 2:
                    ft[i].set_from_ucs4 = &to_uint16;
                    break;
                case 4:
                    ft[i].set_from_ucs4 = &to_uint32;
                    break;
                case 8:
                    ft[i].set_from_ucs4 = &to_uint64;
                    break;
                default:
                    assert(0);
            }
        }
        else if (descr->type_num == NPY_FLOAT) {
            ft[i].set_from_ucs4 = &to_float;
        }
        else if (descr->type_num == NPY_DOUBLE) {
            ft[i].set_from_ucs4 = &to_double;
        }
        else if (descr->type_num == NPY_CFLOAT) {
            ft[i].set_from_ucs4 = &to_cfloat;
        }
        else if (descr->type_num == NPY_CDOUBLE) {
            ft[i].set_from_ucs4 = &to_cdouble;
        }
        else if (descr->type_num == NPY_STRING) {
            ft[i].set_from_ucs4 = &to_string;
        }
        else if (descr->type_num == NPY_UNICODE) {
            ft[i].set_from_ucs4 = &to_unicode;
        }
        else {
            ft[i].set_from_ucs4 = &to_generic;
        }
    }
    return ft;
}
