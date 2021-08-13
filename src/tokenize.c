
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "typedefs.h"
#include "stream.h"

#include "sizes.h"
#include "tokenize.h"
#include "error_types.h"
#include "parser_config.h"


#define ISCOMMENT(c, s, c0, c1) ((c == c0) && ((c1 == 0) || (stream_peek(s) == c1)))

/*
    How parsing quoted fields works:

    For quoting to be activated, the first character of the field
    must be the quote character (after taking into account
    ignore_leading_spaces).  While quoting is active, delimiters
    are treated as regular characters, not delimiters.  Quoting is
    deactivated by the second occurrence of the quote character.  An
    exception is the occurrence of two consecutive quote characters,
    which is treated as a literal occurrence of a single quote character.
    E.g. (with delimiter=',' and quote='"'):
        12.3,"New York, NY","3'2"""
    The second and third fields are `New York, NY` and `3'2"`.

    If a non-delimiter occurs after the closing quote, the quote is
    ignored and parsing continues with quoting deactivated.  Quotes
    that occur while quoting is not activated are not handled specially;
    they become part of the data.
    E.g:
        12.3,"ABC"DEF,XY"Z
    The second and third fields are `ABCDEF` and `XY"Z`.

    Note that the second field of
        12.3,"ABC"   ,4.5
    is `ABC   `.  Currently there is no option to ignore whitespace
    at the end of a field.
*/

/*
 *  tokenize a row of input, with an explicit field delimiter char (sep_char).
 *
 *  word_buffer must point to a block of memory with length word_buffer_size.
 *
 *  The fields are stored in word_buffer, and words is an array of
 *  pointers to the starts of the words parsed so far.
 *
 *  pconfig is in input; *pconfig is a parser_config instance.
 *
 *  Returns an array of char*.  Points to memory malloc'ed here so it
 *  must be freed by the caller.
 *
 *  *p_num_fields and *p_error_type are outputs.
 *  *p_num_fields is the number of fields (i.e. number of tokens) that
 *  were parsed.
 *  *p_error_type is an error code.
 *
 *  Returns NULL for several different conditions:
 *  * Reached EOF before finding *any* data to parse.
 *  * The amount of text copied to word_buffer exceeded the buffer size.
 *  * Failed to parse a single field. This is the condition field_number == 0
 *    that is checked after the main loop.  To do: double check exactly what
 *    can lead to this condition.
 *  * Out of memory: could not allocate the memory to hold the array of
 *    char pointer that the function returns.
 *  * The row has more fields than MAX_NUM_COLUMNS.
 */

char32_t **
tokenize(stream *s, tokenizer_state *ts,
        char32_t *word_buffer, int word_buffer_size,
        parser_config *pconfig, int *p_num_fields, int *p_error_type)
{
    int n;
    char32_t c;
    char32_t *words[MAX_NUM_COLUMNS];
    char32_t *p_word_start, *p_word_end;
    int field_number;
    char32_t **result;

    char32_t cc0 = pconfig->comment[0];
    char32_t cc1 = pconfig->comment[1];
    char32_t sep_char = pconfig->delimiter;
    char32_t quote_char = pconfig->quote;
    bool ignore_leading_spaces = pconfig->ignore_leading_spaces;
    bool ignore_trailing_spaces = pconfig->ignore_trailing_spaces;
    bool allow_embedded_newline = pconfig->allow_embedded_newline;
    int trailing_space_count = 0;
    bool havec;

    *p_error_type = 0;

    havec = true;
    c = stream_fetch(s);
    /* Skip comments or empty lines until a line with content comes */
    while (ISCOMMENT(c, s, cc0, cc1)) {
        stream_skipline(s);
        c = stream_fetch(s);
    }

    if (c == STREAM_EOF) {
        *p_error_type = ERROR_NO_DATA;
        return NULL;
    }

    /* state when a new field begins: */
    int initial_state = TOKENIZE_INIT;
    if (sep_char == '\0') {
        sep_char = ' ';
        initial_state = TOKENIZE_WHITESPACE;
    }
    /* state at beginning of line: */
    int state = initial_state;
    field_number = 0;
    p_word_start = word_buffer;
    p_word_end = p_word_start;

    while (true) {
        if ((p_word_end - word_buffer) >= word_buffer_size) {
            *p_error_type = ERROR_TOO_MANY_CHARS;
            break;
        }
        if (field_number >= MAX_NUM_COLUMNS) {
            *p_error_type = ERROR_TOO_MANY_FIELDS;
            break;
        }
        if (!havec) {
            c = stream_fetch(s);
        }
        else {
            havec = false;
        }

        if (state == TOKENIZE_WHITESPACE) {
            if (c == ' ') {
                continue;
            }
            else {
                state = TOKENIZE_INIT;
            }
        }

        if (state == TOKENIZE_INIT || state == TOKENIZE_UNQUOTED) {
            if (state == TOKENIZE_INIT && c == quote_char) {
                // Opening quote. Switch state to TOKENIZE_QUOTED.
                state = TOKENIZE_QUOTED;
            }
            else if (state == TOKENIZE_INIT && ignore_leading_spaces && c == ' ') {
                // Ignore this leading space.
            }
            else if ((c == sep_char) ||  ISCOMMENT(c, s, cc0, cc1) || (c == '\n') || (c == STREAM_EOF)) {
                if (ISCOMMENT(c, s, cc0, cc1)) {
                    stream_skipline(s);  /* Finished line: forward stream */
                }
                // End of a field.  Save the field, and switch to `initial_state`.
                if (ignore_trailing_spaces && trailing_space_count > 0) {
                    p_word_end -= trailing_space_count;
                }
                *p_word_end = '\0';
                words[field_number] = p_word_start;
                ++field_number;
                ++p_word_end;
                p_word_start = p_word_end;
                if (c != sep_char) {
                    /* Everything not the sep_char ends the line */
                    break;
                }
                trailing_space_count = 0;
                state = initial_state;
            }
            else {
                *p_word_end = c;
                ++p_word_end;
                if (c == ' ') {
                    ++trailing_space_count;
                }
                else {
                    trailing_space_count = 0;
                }
                state = TOKENIZE_UNQUOTED;
            } 
        }
        else if (state == TOKENIZE_QUOTED) {
            if ((c != quote_char && c != '\n' && c != STREAM_EOF) || (c == '\n' && allow_embedded_newline)) {
                *p_word_end = c;
                ++p_word_end;
            }
            else if (c == quote_char && stream_peek(s)==quote_char) {
                // Repeated quote characters; treat the pair as a single quote char.
                *p_word_end = c;
                ++p_word_end;
                // Skip the second double-quote.
                stream_fetch(s);
            }
            else if (c == quote_char) {
                // Closing quote.  Switch state to TOKENIZE_UNQUOTED.
                state = TOKENIZE_UNQUOTED;
                trailing_space_count = 0;
            }
            else {
                // c must be '\n' or STREAM_EOF.
                // If we are here, it means we've reached the end of the file
                // while inside quotes, or the end of the line while inside
                // quotes and 'allow_embedded_newline' is 0.
                // This could be treated as an error, but for now, we'll simply
                // end the field (and the row).
                *p_word_end = '\0';
                words[field_number] = p_word_start;
                ++field_number;
                ++p_word_end;
                p_word_start = p_word_end;
                break;
            }
        }
    }

    if (*p_error_type) {
        return NULL;
    }

    if (field_number == 0) {
        /* XXX Is this the appropriate error type? */
        *p_error_type = ERROR_NO_DATA;
        return NULL;
    }

    *p_num_fields = field_number;
    result = (char32_t **) malloc(sizeof(char32_t *) * field_number);
    if (result == NULL) {
        *p_error_type = ERROR_OUT_OF_MEMORY;
        return NULL;
    }

    for (n = 0; n < field_number; ++n) {
        result[n] = words[n];
    }

    return result;
}
