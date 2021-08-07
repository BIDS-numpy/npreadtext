
#ifndef _FIELD_TYPES_H_
#define _FIELD_TYPES_H_

#include <stdint.h>
#include <stdbool.h>
#define PY_ARRAY_UNIQUE_SYMBOL npreadtext_ARRAY_API
#include <numpy/ndarraytypes.h>


typedef struct _field_type {
    // typecode:
    //     * : undefined field type (during auto-discovery)
    //     i : integer (byte to long long)
    //     u : unsigned integer (ubyte to unsigned long long)
    //     f : floating point (or double) -- not float128, as it can't be
    //         handled by the builtin parser (we ignore it here).
    //     c : complex (64bit or 128 bit).
    //     S : character string (1 character == 1 byte)
    //     U : Unicode string (32 bit codepoints)
    //     x : generic dtype.
    char typecode;
    /* Whether the simple(!) typecode descriptor needs swapping */
    bool swap;

    // itemsize:
    //   Size of field, in bytes.  In theory this would only be
    //   needed for the 'S' or 'U' type codes, but it is expected to be
    //   correctly filled in for all the types.
    size_t itemsize;

    /* The original NumPy descriptor */
    PyArray_Descr *descr;
} field_type;


void
field_types_clear(int num_field_types, field_type *ft);

field_type *
field_types_create(int num_field_types, PyArray_Descr *dtypes[]);

void
field_types_fprintf(FILE *out, int num_field_types, const field_type *ft);

bool
field_types_is_homogeneous(int num_field_types, const field_type *ft);

int32_t
field_types_total_size(int num_field_types, const field_type *ft);

int
field_types_grow(int new_num_field_types, int num_field_types, field_type **ft);

int
field_types_init_descriptors(int num_field_types, field_type *ft);

PyArray_Descr *
field_types_to_descr(int num_fields, field_type *ft);

char *
typecode_to_str(char typecode);

#endif
