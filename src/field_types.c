
//
// Pure C -- no Python API.
// With TESTMAIN defined, this file has a main() that runs some "tests"
// (actually it just prints some stuff, to be verified by the user).
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "field_types.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include <numpy/arrayobject.h>


void
field_types_clear(int num_field_types, field_type *ft) {
    for (int i = 0; i < num_field_types; i++) {
        Py_XDECREF(ft[i].descr);
        ft[i].descr = NULL;
    }
}

field_type *
field_types_create(int num_field_types, PyArray_Descr *dtypes[])
{
    field_type *ft;

    ft = malloc(num_field_types * sizeof(field_type));
    if (ft == NULL) {
        return NULL;
    }
    for (int i = 0; i < num_field_types; ++i) {
        PyArray_Descr *descr = dtypes[i];
        if (PyDataType_ISSIGNED(descr)) {
            ft[i].typecode = 'i';
        }
        else if (PyDataType_ISUNSIGNED(descr)) {
            ft[i].typecode = 'u';
        }
        else if (PyDataType_ISFLOAT(descr) && descr->elsize <= 8) {
            ft[i].typecode = 'f';
        }
        else if (PyDataType_ISCOMPLEX(descr) && descr->elsize <= 16) {
            ft[i].typecode = 'c';
        }
        else if (descr->type_num == NPY_STRING) {
            ft[i].typecode = 'S';
        }
        else if (descr->type_num == NPY_UNICODE) {
            ft[i].typecode = 'U';
        }
        else {
            ft[i].typecode = 'x';
        }
        ft[i].itemsize = descr->elsize;
        Py_INCREF(descr);
        ft[i].descr = descr;
        ft[i].swap = !PyArray_ISNBO(descr->byteorder);
    }
    return ft;
}

void
field_types_fprintf(FILE *out, int num_field_types, const field_type *ft)
{
    for (int i = 0; i < num_field_types; ++i) {
        fprintf(out, "ft[%d].typecode = %c, .itemsize = %lu\n",
                     i, ft[i].typecode, (unsigned long)ft[i].itemsize);
    }
}


bool
field_types_is_homogeneous(int num_field_types, const field_type *ft)
{
    bool homogeneous = true;
    for (int k = 1; k < num_field_types; ++k) {
        if ((ft[k].typecode != ft[0].typecode) ||
                (ft[k].itemsize != ft[0].itemsize)) {
            homogeneous = false;
            break;
        }
    }
    return homogeneous;
}


int32_t
field_types_total_size(int num_field_types, const field_type *ft)
{
    int32_t size = 0;
    for (int k = 0; k < num_field_types; ++k) {
        size += ft[k].itemsize;
    }
    return size;
}


//
// *ft must be a pointer to field_type that is either NULL or
// previously malloc'ed.
// The function assumes that new_num_field_types > num_field_types.
//
int
field_types_grow(int new_num_field_types, int num_field_types, field_type **ft)
{
    size_t nbytes;
    field_type *new_ft;

    nbytes = new_num_field_types * sizeof(field_type);
    new_ft = (field_type *) realloc(*ft, nbytes);
    if (new_ft == NULL) {
        free(*ft);
        *ft = NULL;
        return -1;
    }
    for (int k = num_field_types; k < new_num_field_types; ++k) {
        new_ft[k].typecode = '*';
        new_ft[k].itemsize = 0;
        new_ft[k].descr = NULL;
        new_ft[k].swap = false;  /* normally never swap */
    }
    *ft = new_ft;

    return 0;
}


/*
 * Convert the discovered dtypes to descriptors.  This function cleans up
 * after itself on error.
 *
 * TODO: We are assuming here sane systems with 1, 2, 4, and 8 byte sized
 *       integers.  But NumPy relies more on the C levels.
 */
int
field_types_init_descriptors(int num_field_types, field_type *ft)
{
    for (int i = 0; i < num_field_types; i++) {
        int typenum = -1;

        /* Strings have adaptable length, so handle it specifically. */
        if (ft[i].typecode == 'S' || ft[i].typecode == 'U') {
            typenum = ft[i].typecode == 'S' ? NPY_STRING : NPY_UNICODE;
            ft[i].descr = PyArray_DescrNewFromType(typenum);
            if (ft[i].descr == NULL) {
                field_types_clear(num_field_types, ft);
                return -1;
            }
            ft[i].descr->elsize = ft[i].itemsize;
            continue;
        }

        if (ft[i].typecode == 'i') {
            size_t itemsize = ft[i].itemsize;
            switch (itemsize) {
                case 1:
                    typenum = NPY_INT8;
                    break;
                case 2:
                    typenum = NPY_INT16;
                    break;
                case 4:
                    typenum = NPY_INT32;
                    break;
                case 8:
                    typenum = NPY_INT64;
                    break;
            }
        }
        else if (ft[i].typecode == 'u') {
            size_t itemsize = ft[i].itemsize;
            switch (itemsize) {
                case 1:
                    typenum = NPY_UINT8;
                    break;
                case 2:
                    typenum = NPY_UINT16;
                    break;
                case 4:
                    typenum = NPY_UINT32;
                    break;
                case 8:
                    typenum = NPY_UINT64;
                    break;
            }
        }
        else if (ft[i].typecode == 'f') {
            typenum = NPY_DOUBLE;
        }
        else if (ft[i].typecode == 'c') {
            typenum = NPY_CDOUBLE;
        }
        else if (ft[i].typecode == '*') {
            typenum = NPY_DOUBLE;  /* use the default */
        }
        else {
            /* will error below, but add assert for debugging */
            assert(0);
        }

        ft[i].descr = PyArray_DescrFromType(typenum);
        if (ft[i].descr == NULL) {
            /* can't actually fail, but... */
            field_types_clear(num_field_types, ft);
            return -1;
        }
    }
    return 0;
}


PyArray_Descr *
field_types_to_descr(int num_fields, field_type *ft)
{
    PyArray_Descr *result = NULL;
    PyObject *dtype_list = NULL, *empty_string = NULL;

    empty_string = PyUnicode_FromString("");
    if (empty_string == NULL) {
        goto finish;
    }
    dtype_list = PyList_New(0);
    if (dtype_list == NULL) {
        goto finish;
    }

    for (int i = 0; i < num_fields; i++) {
        PyObject *tup = PyTuple_Pack(2, empty_string, ft[i].descr);
        if (tup == NULL) {
            goto finish;
        }
        int res = PyList_Append(dtype_list, tup);
        Py_DECREF(tup);
        if (res < 0) {
            goto finish;
        }
    }
    PyArray_DescrConverter(dtype_list, &result);

  finish:
    Py_XDECREF(empty_string);
    Py_XDECREF(dtype_list);
    return result;
}

#ifdef TESTMAIN

void
show_field_types(int num_fields, field_type *ft)
{
    field_types_fprintf(stdout, num_fields, ft);

    bool homogeneous = field_types_is_homogeneous(num_fields, ft);
    printf("homogeneous = %s\n", homogeneous ? "true" : "false");

    int32_t total_size = field_types_total_size(num_fields, ft);
    printf("total_size = %lu\n", (unsigned long)total_size);

    char *dtypestr = field_types_build_str(num_fields, NULL, homogeneous, ft);
    printf("dtypestr = '%s'\n", dtypestr);

    free(dtypestr);
}


int
main(int argc, char *argv[])
{
    char *codes = "ffHHSU";
    int32_t sizes[] = {8, 8, 2, 2, 4, 48};
    int num_fields = sizeof(sizes) / sizeof(sizes[0]);

    field_type *ft = field_types_create(num_fields, codes, sizes);
    printf("ft:\n");
    show_field_types(num_fields, ft);

    int status = field_types_grow(num_fields + 2, num_fields, &ft);
    ft[num_fields].typecode = 'b';
    ft[num_fields].itemsize = 1;
    ++num_fields;
    ft[num_fields].typecode = 'f';
    ft[num_fields].itemsize = 4;
    ++num_fields;

    printf("\n");
    printf("ft, after grow:\n");
    show_field_types(num_fields, ft);

    free(ft);

    return 0;
}

#endif
