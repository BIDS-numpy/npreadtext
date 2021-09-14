
#ifndef _PARSER_CONFIG_H_
#define _PARSER_CONFIG_H_

#include <stdint.h>
#include <stdbool.h>


typedef struct _parser_config {
    /*
     *  Field delimiter character.
     *  Typically ',', ' ', '\t', or '\0'.
     */
    Py_UCS4 delimiter;

    /*
     *  Character used to quote fields.
     *  Typically '"' or "'".  To disable quoting we set this to UINT_MAX
     *  (which is not a valid unicode character and thus cannot occur in the
     *  file; the same is used for all other characters if necessary).
     */
    Py_UCS4 quote;

    /*
     *  Ignore whitespace at the beginning of a field (outside/before quotes).
     *  Is implicitly always set if we split on any whitespace.
     */
    bool ignore_leading_whitespace;

    /*
     * If true, the delimiter is ignored and any unicode whitespace is used
     * for splitting (same as `string.split()` in Python).
     */
    bool delimiter_is_whitespace;

    /*
     *  Character(s) that indicates the start of a comment.
     *  Typically '#', '%' or ';'.
     *  When encountered in a line and not inside quotes, all character
     *  from the comment character(s) to the end of the line are ignored.
     */
    Py_UCS4 comment;

    /*
     *  A boolean value (0 or 1).  If 1, quoted fields may span
     *  more than one line.  For example, the following
     *      100, 200, "FOO
     *      BAR"
     *  is one "row", containing three fields: 100, 200 and "FOO\nBAR".
     *  If 0, the parser considers an unclosed quote to be an error. (XXX Check!)
     */
    bool allow_embedded_newline;

    /*
     *  The decimal point character.
     *  Most commonly '.', but ',' is sometimes used.
     */
    Py_UCS4 decimal;

    /*
     *  The character used to indicate the exponent in scientific notation.
     *  Typically 'E' or 'e', but 'D' (or 'd') are sometimes used (mainly in
     *  Fortran code).  When parsing, the case is ignored.
     */
    Py_UCS4 sci;

    /*
     *  The imaginary unit character. Default is `j`.
     */
    Py_UCS4 imaginary_unit;

     /*
      *  If true, when an integer dtype is given, the field is allowed
      *  to contain a floating point value.  It will be cast to the
      *  integer type.
      */
     bool allow_float_for_int;
     /*
      * Data should be encoded as `latin1` when using python converter
      * (implementing `loadtxt` default Python 2 compatibility mode).
      * The c byte converter is used when the user requested `dtype="S"`.
      * In this case we go via `dtype=object`, however, loadtxt allows latin1
      * while normal object to string casts only accept ASCII, so it ensures
      * that that the object array already contains bytes and not strings.
      */
     bool python_byte_converters;
     bool c_byte_converters;
} parser_config;

parser_config
default_parser_config(void);

#endif
