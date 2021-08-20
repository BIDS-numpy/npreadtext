#ifndef STR_TO_H
#define STR_TO_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

#include "typedefs.h"
#include "str_to.h"


#include "typedefs.h"
#include "numpy/ndarraytypes.h"


#define ERROR_OK             0
#define ERROR_NO_DIGITS      1
#define ERROR_OVERFLOW       2
#define ERROR_INVALID_CHARS  3
#define ERROR_MINUS_SIGN     4

/*
 * The following two string conversion functions are largely equivalent
 * in Pandas.  They are in the header file here, to ensure they can be easily
 * inline in the other function.
 */

/*
 *  On success, *error is zero.
 *  If the conversion fails, *error is nonzero, and the return value is 0.
 */
int64_t NPY_INLINE
str_to_int64(
        const char32_t *p_item, int64_t int_min, int64_t int_max, int *error)
{
    const char32_t *p = (const char32_t *) p_item;
    bool isneg = 0;
    int64_t number = 0;
    int d;

    // Skip leading spaces.
    while (isspace(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        isneg = true;
        ++p;
    }
    else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit(*p)) {
        // Error...
        *error = ERROR_NO_DIGITS;
        return 0;
    }

    if (isneg) {
        // If number is greater than pre_min, at least one more digit
        // can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // Process the digits.
        d = *p;
        while (isdigit(d)) {
            if ((number > pre_min) || ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                number = number * 10 - (d - '0');
                d = *++p;
            }
            else {
                *error = ERROR_OVERFLOW;
                return 0;
            }
        }
    }
    else {
        // If number is less than pre_max, at least one more digit
        // can be processed without overflowing.
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;

        //printf("pre_max = %lld  dig_pre_max = %d\n", pre_max, dig_pre_max);

        // Process the digits.
        d = *p;
        while (isdigit(d)) {
            if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                d = *++p;
            }
            else {
                *error = ERROR_OVERFLOW;
                return 0;
            }
        }
    }

    // Skip trailing spaces.
    while (isspace(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (*p) {
        *error = ERROR_INVALID_CHARS;
        return 0;
    }

    *error = 0;
    return number;
}

/*
 *  On success, *error is zero.
 *  If the conversion fails, *error is nonzero, and the return value is 0.
 */
uint64_t NPY_INLINE
str_to_uint64(const char32_t *p_item, uint64_t uint_max, int *error)
{
    const char32_t *p = (const char32_t *) p_item;
    uint64_t number = 0;
    int d;

    // Skip leading spaces.
    while (isspace(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        *error = ERROR_MINUS_SIGN;
        return 0;
    }
    if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit(*p)) {
        // Error...
        *error = ERROR_NO_DIGITS;
        return 0;
    }

    // If number is less than pre_max, at least one more digit
    // can be processed without overflowing.
    uint64_t pre_max = uint_max / 10;
    int dig_pre_max = uint_max % 10;

    // Process the digits.
    d = *p;
    while (isdigit(d)) {
        if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
            number = number * 10 + (d - '0');
            d = *++p;
        }
        else {
            *error = ERROR_OVERFLOW;
            return 0;
        }
    }

    // Skip trailing spaces.
    while (isspace(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (*p) {
        *error = ERROR_INVALID_CHARS;
        return 0;
    }

    *error = 0;
    return number;
}


#endif
