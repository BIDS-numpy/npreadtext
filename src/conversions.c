
#include <Python.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <stdbool.h>

#include "conversions.h"
#include "str_to_int.h"


/*
 * Coercion to boolean is done via integer right now.
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
 * In order to not pack a whole copy of a floating point parser, we copy the
 * result into ascii and call the Python one.  Float parsing isn't super quick
 * so this is not terrible, but avoiding it would speed up things.
 *
 * Also note that parsing the first float of a complex will copy the whole
 * string to ascii rather than just the first part.
 * TODO: A tweak of the break might be a simple mitigation there.
 *
 * @param str The UCS4 string to parse
 * @param end Pointer to the end of the string
 * @param skip_trailing_whitespace If false does not skip trailing whitespace
 *        (used by the complex parser).
 * @param result Output stored as double value.
 */
static NPY_INLINE int
double_from_ucs4(
        const Py_UCS4 *str, const Py_UCS4 *end,
        bool skip_trailing_whitespace, double *result, const Py_UCS4 **p_end)
{
    /* skip leading whitespace */
    while (Py_UNICODE_ISSPACE(*str)) {
        str++;
    }
    if (str == end) {
        return -1;  /* empty or only whitespace: not a floating point number */
    }

    /* We convert to ASCII for the Python parser, use stack if small: */
    char stack_buf[128];
    char *heap_buf = NULL;
    char *ascii = stack_buf;

    size_t str_len = end - str;
    if (str_len > 128) {
        heap_buf = PyMem_MALLOC(str_len);
        ascii = heap_buf;
    }
    char *c = ascii;
    for (; str < end; str++, c++) {
        if (NPY_UNLIKELY(*str >= 128)) {
            break;  /* the following cannot be a number anymore */
        }
        *c = (char)(*str);
    }
    *c = '\0';

    char *end_parsed;
    *result = PyOS_string_to_double(ascii, &end_parsed, NULL);
    /* Rewind `end` to the first UCS4 character not parsed: */
    end = end - (c - end_parsed);

    PyMem_FREE(heap_buf);

    if (*result == -1. && PyErr_Occurred()) {
        return -1;
    }

    if (skip_trailing_whitespace) {
        /* and then skip any remainig whitespace: */
        while (Py_UNICODE_ISSPACE(*end)) {
            end++;
        }
    }
    *p_end = end;
    return 0;
}

/*
 *  `item` must be the nul-terminated string that is to be
 *  converted to a double.
 *
 *  To be successful, to_double() must use *all* the characters
 *  in `item`.  E.g. "1.q25" will fail.  Leading and trailing 
 *  spaces are allowed.
 */
int
to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    double double_val;
    const Py_UCS4 *p_end;
    if (double_from_ucs4(str, end, true, &double_val, &p_end) < 0) {
        return -1;
    }
    if (p_end != end) {
        return -1;
    }

    float val = double_val;
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
    double val;
    const Py_UCS4 *p_end;
    if (double_from_ucs4(str, end, true, &val, &p_end) < 0) {
        return -1;
    }
    if (p_end != end) {
        return -1;
    }

    memcpy(dataptr, &val, sizeof(double));
    if (!PyArray_ISNBO(descr->byteorder)) {
        descr->f->copyswap(dataptr, dataptr, 1, NULL);
    }
    return 0;
}


static bool
to_complex_int(
        const Py_UCS4 *item, const Py_UCS4 *token_end,
        double *p_real, double *p_imag,
        Py_UCS4 imaginary_unit, bool allow_parens)
{
    const Py_UCS4 *p_end;
    bool unmatched_opening_paren = false;

    /* Remove whitespace before the possibly leading '(' */
    while (Py_UNICODE_ISSPACE(*item)) {
        ++item;
    }
    if (allow_parens && (*item == '(')) {
        unmatched_opening_paren = true;
        ++item;
    }
    if (double_from_ucs4(item, token_end, false, p_real, &p_end) < 0) {
        return false;
    }
    if (p_end == token_end) {
        // No imaginary part in the string (e.g. "3.5")
        *p_imag = 0.0;
        return !unmatched_opening_paren;
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
        if (double_from_ucs4(p_end, token_end, false, p_imag, &p_end) < 0) {
            return false;
        }
        if (*p_end != imaginary_unit) {
            return false;
        }
        ++p_end;
        if (unmatched_opening_paren && (*p_end == ')')) {
            ++p_end;
            unmatched_opening_paren = false;
        }
    }
    while (Py_UNICODE_ISSPACE(*p_end)) {
        ++p_end;
    }
    return p_end == token_end;
}


int
to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    double real;
    double imag;

    bool success = to_complex_int(
            str, end, &real, &imag,
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
    double real;
    double imag;

    bool success = to_complex_int(
            str, end, &real, &imag,
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