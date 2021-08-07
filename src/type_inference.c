#define _XOPEN_SOURCE

#include <stdio.h>
#include <time.h>
#include <stdbool.h>

#include "typedefs.h"
#include "str_to.h"
#include "conversions.h"

#define ALLOW_PARENS true

/*

Some use cases:

Suppose the file contains:
-----------------------------
100,1.2,,,
200,1.4,,,
300,1.8,,,
400,2.0,19,-1,5.0
500,2.5,21,-3,7.5
-----------------------------
Then the last three columns should be classified as
'B', 'b', 'd', respectively, and the data for the
first three rows for these columns should be treated
as missing data.

*/

/*
 *  char classify_type(char *field, char decimal, char sci, int64_t *i, uint64_t *u, char *datetime_fmt)
 *
 *  Try to parse the field, in the following order:
 *      unsigned int  ('Q')
 *      int ('q')
 *      floating point ('d')
 *      complex ('z')
 *  If those all fail, the field type is called 'S'.
 *
 *  If the classification is 'Q' or 'q', the value
 *  of the integer is stored in *u or *i, resp.
 *
 *  prev_type == '*' means there is no previous sample from this column.
 *
 *  XXX How should a fields of spaces be classified?
 *      What about an empty field, ""?
 *      In isolation, a blank field could be classified '*'.  How should
 *      blanks in a column be used?  It seems that, if the prev_type is '*',
 *      it means we don't what the column is.  If a blank field is encountered
 *      when prev_type != '*', the field should stay classified as prev_type.
 *      When we are using prev_type, the problem we are solving is actually
 *      to classify a column, not just a single field. 
 */

char
classify_type(char32_t *field,
        char32_t decimal, char32_t sci, char32_t imaginary_unit,
        int64_t *i, uint64_t *u, char prev_type)
{
    int error = 0;
    int success;
    double real, imag;
    //struct tm tm;

    switch (prev_type) {
        case '*':
        case 'u':
        case 'i':
            *u = str_to_uint64(field, UINT64_MAX, &error);
            if (error == 0) {
                return prev_type == 'i' ? 'i' : 'u';  /* retain if signed */
            }
            if (error == ERROR_MINUS_SIGN) {
                *i = str_to_int64(field, INT64_MIN, INT64_MAX, &error);
                if (error == 0) {
                    return 'i';
                }
            }
            /*@fallthrough@*/
        case 'f':
            success = to_double(field, &real, sci, decimal);
            if (success) {
                return 'f';
            }
            /*@fallthrough@*/
        case 'c':
            success = to_complex(field, &real, &imag, sci, decimal, imaginary_unit,
                                 ALLOW_PARENS);
            if (success) {
                return 'c';
            }
        /* TODO: We may want try dates. Some dates can be integers? */
    }
    while (*field == ' ') {
        ++field;
    }
    if (!*field) {
        /* All spaces, so return prev_type */
        return prev_type;
    }
    return 'S';
}

/*
 *  char type_for_integer_range(int64_t imin, uint64_t umax)
 *
 *  Determine an appropriate type character for the given
 *  range of integers.  The function assumes imin <= 0.
 *  If imin == 0, the return value will be one of
 *      'B': 8 bit unsigned,
 *      'H': 16 bit unsigned,
 *      'I': 32 bit unsigned,
 *      'Q': 64 bit unsigned.
 *  If imin < 0, the possible return values
 *  are
 *      'b': 8 bit signed,
 *      'h': 16 bit signed,
 *      'i': 32 bit signed,
 *      'q': 64 bit signed
 *      'd': floating point double precision
 *
 */

void
update_type_for_integer_range(
        char *type, size_t *itemsize, int64_t imin, uint64_t umax)
{
    if (*type == 'u') {
        /* unsigned int type */
        if (umax <= UINT8_MAX) {
            *itemsize = 1;
        }
        else if (umax <= UINT16_MAX) {
            *itemsize = 2;
        }
        else if (umax <= UINT32_MAX) {
            *itemsize = 4;
        }
        else {
            *itemsize = 8;
        }
    }
    else {
        /* int type */
        if (imin >= INT8_MIN && umax <= INT8_MAX) {
            *itemsize = 1;
        }
        else if (imin >= INT16_MIN && umax <= INT16_MAX) {
            *itemsize = 2;
        }
        else if (imin >= INT32_MIN && umax <= INT32_MAX) {
            *itemsize = 4;
        }
        else if (umax <= INT64_MAX) {
            *itemsize = 8;
        }
        else {
            /* 
             *  imin < 0 and the largest value exceeds INT64_MAX, so this
             * range can not be represented with an integer format.
             *  We'll have to convert these to floating point.
             */
            *type = 'f';
            *itemsize = 8;
        }
    }
}
