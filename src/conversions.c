
#include <Python.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>

#include "conversions.h"
#include "str_to_int.h"


double
_Py_dg_strtod_modified(
        const Py_UCS4 *s00, Py_UCS4 **se, int *error,
        Py_UCS4 decimal, Py_UCS4 sci, bool skip_trailing);


/*
 * Coercion to boolean is done via integer right now.
 * TODO: Like the integer code, does not handle embedded \0 characters!
 */
int
to_bool(PyArray_Descr *NPY_UNUSED(descr),
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    int64_t res;
    if (str_to_int64(str, end, INT64_MIN, INT64_MAX, &res) < 0) {
        return -1;
    }
    *dataptr = (res != 0);
    return 0;
}


/*
 *  `item` must be the nul-terminated string that is to be
 *  converted to a double.
 *
 *  To be successful, to_double() must use *all* the characters
 *  in `item`.  E.g. "1.q25" will fail.  Leading and trailing 
 *  spaces are allowed.
 *
 *  `sci` is the scientific notation exponent character, usually
 *  either 'E' or 'D'.  Case is ignored.
 *
 *  `decimal` is the decimal point character, usually either
 *  '.' or ','.
 *
 */
int
to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    int error;
    Py_UCS4 *p_end;

    float val = (float)_Py_dg_strtod_modified(
            str, &p_end, &error, pconfig->decimal, pconfig->sci, true);

    if (error != 0 || p_end != end) {
        return -1;
    }
    memcpy(dataptr, &val, sizeof(float));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


int
to_double(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    int error;
    Py_UCS4 *p_end;

    double val = _Py_dg_strtod_modified(
            str, &p_end, &error, pconfig->decimal, pconfig->sci, true);

    if (error != 0 || p_end != end) {
        return -1;
    }
    memcpy(dataptr, &val, sizeof(double));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


static bool
to_complex_raw(
        const Py_UCS4 *item, double *p_real, double *p_imag,
        Py_UCS4 sci, Py_UCS4 decimal, Py_UCS4 imaginary_unit,
        bool allow_parens)
{
    Py_UCS4 *p_end;
    int error;
    bool unmatched_opening_paren = false;

    if (allow_parens && (*item == '(')) {
        unmatched_opening_paren = true;
        ++item;
    }
    *p_real = _Py_dg_strtod_modified(item, &p_end, &error, decimal, sci, false);
    if (*p_end == '\0') {
        // No imaginary part in the string (e.g. "3.5")
        *p_imag = 0.0;
        return (error == 0) && (!unmatched_opening_paren);
    }
    if (*p_end == imaginary_unit) {
        // Pure imaginary part only (e.g "1.5j")
        *p_imag = *p_real;
        *p_real = 0.0;
        ++p_end;
        if (unmatched_opening_paren && (*p_end == ')')) {
            ++p_end;
            unmatched_opening_paren = false;
        }
    }
    else if (unmatched_opening_paren && (*p_end == ')')) {
        *p_imag = 0.0;
        ++p_end;
        unmatched_opening_paren = false;
    }
    else {
        if (*p_end == '+') {
            ++p_end;
        }

        *p_imag = _Py_dg_strtod_modified(p_end, &p_end, &error, decimal, sci, false);
        if (error || (*p_end != imaginary_unit)) {
            return false;
        }
        ++p_end;
        if (unmatched_opening_paren && (*p_end == ')')) {
            ++p_end;
            unmatched_opening_paren = false;
        }
    }
    while(*p_end == ' ') {
        ++p_end;
    }
    return *p_end == '\0';
}


int
to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    double real;
    double imag;

    // TODO: This should check the end pointer (needs fixing in to_complex_int)
    bool success = to_complex_raw(
            str, &real, &imag,
            pconfig->sci, pconfig->decimal,
            pconfig->imaginary_unit, true);

    if (!success) {
        return -1;
    }
    npy_complex64 val = {real, imag};
    memcpy(dataptr, &val, sizeof(npy_complex64));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


int
to_cdouble(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    if (*str == '\0') {
        /* can't parse NUL, this field is probably just empty. */
        return -1;
    }

    double real;
    double imag;

    // TODO: This should check the end pointer (needs fixing in to_complex_int)
    bool success = to_complex_raw(
            str, &real, &imag,
            pconfig->sci, pconfig->decimal,
            pconfig->imaginary_unit, true);

    if (!success) {
        return -1;
    }
    npy_complex128 val = {real, imag};
    memcpy(dataptr, &val, sizeof(npy_complex128));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


/*
 * String and unicode conversion functions.
 */
int
to_string(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused)
{
    const Py_UCS4* c = str;
    size_t length = descr->elsize;

    for (size_t i = 0; i < length; i++) {
        if (c < end) {
            /*
             * loadtxt assumed latin1, which is compatible with UCS1 (first
             * 256 unicode characters).
             */
            if (NPY_UNLIKELY(*c > 255)) {
                /* TODO: Was UnicodeDecodeError, is unspecific error good? */
                return -1;
            }
            dataptr[i] = (Py_UCS1)(*c);
            c++;
        }
        else {
            dataptr[i] = '\0';
        }
    }
    return 0;
}


int
to_unicode(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused)
{
    size_t length = descr->elsize / 4;

    if (length <= (size_t)(end - str)) {
        memcpy(dataptr, str, length * 4);
    }
    else {
        size_t given_len = end - str;
        memcpy(dataptr, str, given_len * 4);
        memset(dataptr + given_len * 4, '\0', (length -given_len) * 4);
    }

    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}



/*
 * Convert functions helper for the generic converter.
 */
static PyObject *
call_converter_function(
        PyObject *func, const Py_UCS4 *str, size_t length, bool byte_converters)
{
    PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, str, length);
    if (s == NULL) {
        return s;
    }
    if (byte_converters) {
        Py_SETREF(s, PyUnicode_AsEncodedString(s, "latin1", NULL));
        if (s == NULL) {
            return NULL;
        }
    }
    if (func == NULL) {
        return s;
    }
    PyObject *result = PyObject_CallFunctionObjArgs(func, s, NULL);
    Py_DECREF(s);
    return result;
}


/*
 * Defines liberated from NumPy's, only used for the PyArray_Pack hack!
 * TODO: Remove!
 */
#if PY_VERSION_HEX < 0x030900a4
    /* Introduced in https://github.com/python/cpython/commit/d2ec81a8c99796b51fb8c49b77a7fe369863226f */
    #define Py_SET_TYPE(obj, type) ((Py_TYPE(obj) = (type)), (void)0)
    /* Introduced in https://github.com/python/cpython/commit/c86a11221df7e37da389f9c6ce6e47ea22dc44ff */
    #define Py_SET_REFCNT(obj, refcnt) ((Py_REFCNT(obj) = (refcnt)), (void)0)
#endif

int
to_generic_with_converter(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *config, PyObject *func)
{
    bool use_byte_converter;
    if (func == NULL) {
        use_byte_converter = config->c_byte_converters;
    }
    else {
        use_byte_converter = config->python_byte_converters;
    }
    /* Converts to unicode and calls custom converter (if set) */
    PyObject *converted = call_converter_function(
            func, str, (size_t)(end - str), use_byte_converter);
    if (converted == NULL) {
        return -1;
    }
    /* TODO: Dangerous semi-copy from PyArray_Pack which this
     *       should use, but cannot (it is not yet public).
     *       This will get some casts wrong (unlike PyArray_Pack),
     *       and like it (currently) does necessarily handle an
     *       array return correctly (but maybe that is fine).
     */
    PyArrayObject_fields arr_fields = {
            .flags = NPY_ARRAY_WRITEABLE,  /* assume array is not behaved. */
    };
    Py_SET_TYPE(&arr_fields, &PyArray_Type);
    Py_SET_REFCNT(&arr_fields, 1);
    arr_fields.descr = descr;
    int res = descr->f->setitem(converted, dataptr, &arr_fields);
    Py_DECREF(converted);
    if (res < 0) {
        return -1;
    }
    return 0;
}


int
to_generic(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *config)
{
    return to_generic_with_converter(descr, str, end, dataptr, config, NULL);
}