
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "typedefs.h"
#include "stream.h"

#include "sizes.h"
#include "tokenize.h"
#include "error_types.h"
#include "parser_config.h"

#include "numpy/ndarraytypes.h"

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


static size_t
next_size(size_t size) {
    return ((size_t)size + 3) & ~(size_t)3;
}


static int
copy_to_field_buffer(tokenizer_state *ts,
        char32_t *chunk_start, char32_t *chunk_end,
        bool started_saving_word,
        size_t *word_start, size_t *word_length)
{
    size_t chunk_length = chunk_end - chunk_start;

    size_t size = chunk_length + ts->field_buffer_pos + 1;

    if (NPY_UNLIKELY(ts->field_buffer_length < size)) {
        size = next_size(size);
        char32_t *new = PyMem_Realloc(ts->field_buffer, size * sizeof(char32_t));
        if (new == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        ts->field_buffer_length = size;
        ts->field_buffer = new;
    }

    if (!started_saving_word) {
        *word_start = ts->field_buffer_pos;
        *word_length = 0;
    }

    memcpy(ts->field_buffer + ts->field_buffer_pos, chunk_start,
           chunk_length * sizeof(char32_t));
    ts->field_buffer_pos += chunk_length;
    *word_length += chunk_length;
    /* always ensure we end with NUL */
    ts->field_buffer[ts->field_buffer_pos] = '\0';
    return 0;
}


int
add_field(tokenizer_state *ts,
        size_t word_start, size_t word_length, bool is_quoted)
{
    ts->field_buffer_pos += 1;  /* The field is done, so advance for next one */
    size_t size = ts->num_fields + 1;
    if (NPY_UNLIKELY(size > ts->fields_size)) {
        size = next_size(size);
        field_info *fields = PyMem_Realloc(ts->fields, size * sizeof(*fields));
        if (fields == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        ts->fields = fields;
        ts->fields_size = size;
    }

    ts->fields[ts->num_fields].offset = word_start;
    ts->fields[ts->num_fields].length = word_length;
    ts->fields[ts->num_fields].quoted = is_quoted;
    ts->num_fields += 1;
    return 0;
}


/*
 * This version now always copies the full "row" (all tokens).  This makes
 * two things easier:
 * 1. It means that every word is guaranteed to be followed by a NUL character
 *    (although it can include one as well).
 * 2. In the usecols case we can sniff the first row easier by parsing it
 *    fully.
 *
 * The tokenizer could grow the ability to skip fields and check the
 * maximum number of fields when known.
 *
 * Unlike other tokenizers, this one tries to work in chunks and copies
 * data to words only when it it has to.  The hope is that this makes multiple
 * light-weight loops rather than a single heavy one, to allow e.g. quickly
 * scanning for the end of a field.
 */
int
tokenize(stream *s, tokenizer_state *ts, parser_config *const config)
{
    char32_t *chunk_start = NULL;
    char32_t *chunk_end;

    size_t word_start;
    size_t word_length;

    /* Reset to start of buffer */
    ts->field_buffer_pos = 0;

    char32_t *pos = ts->pos;

    ts->num_fields = 0;
    bool is_quoted = false;
    bool started_saving_word = false;

    int res = 0;

    while (1) {
        if (NPY_UNLIKELY(pos >= ts->end)) {
            /* fetch new data */
            ts->buf_state = stream_nextbuf(s, &ts->pos, &ts->end);
            if (ts->buf_state < 0) {
                return -1;
            }
            else if (ts->pos == ts->end) {
                if (ts->buf_state != BUFFER_IS_FILEEND) {
                    PyErr_SetString(PyExc_RuntimeError,
                            "Reader returned an empty buffer, "
                            "but file did not end.");
                    return -1;
                }
                if (ts->state & TOKENIZE_OUTSIDE_FIELD) {
                    res = 1;
                    goto finish;
                }
                /* We probably still need to append the last field */
                ts->state = TOKENIZE_FINALIZE_FILE;
            }
            pos = ts->pos;
        }

        switch (ts->state) {
            case TOKENIZE_UNQUOTED:
                chunk_start = pos;
                for (; pos < ts->end; pos++) {
                    if (*pos == '\r' || *pos == '\n') {
                        ts->state = TOKENIZE_EAT_CRLF;
                        break;
                    }
                    else if (*pos == config->delimiter) {
                        ts->state = TOKENIZE_INIT;
                        break;
                    }
                    else if (*pos == config->comment[0]) {
                        if (config->comment[1] != '\0') {
                            ts->state = TOKENIZE_CHECK_COMMENT;
                            break;
                        }
                        else {
                            ts->state = TOKENIZE_FINALIZE_LINE;
                            break;
                        }
                    }
                }
                chunk_end = pos;
                pos++;
                break;

            case TOKENIZE_QUOTED:
                chunk_start = pos;
                for (; pos < ts->end; pos++) {
                    if (!config->allow_embedded_newline && (
                                *pos == '\r' || *pos == '\n')) {
                        ts->state = TOKENIZE_EAT_CRLF;
                        break;
                    }
                    else if (*pos != config->quote) {
                        /* inside the field, nothing to do. */
                    }
                    else {
                        ts->state = TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE;
                        break;
                    }
                }
                chunk_end = pos;
                pos++;
                break;

            case TOKENIZE_WHITESPACE:
                for (; pos < ts->end; pos++) {
                    if (*pos != ' ') {
                        ts->state = TOKENIZE_INIT;
                        break;
                    }
                }
                break;

            case TOKENIZE_INIT:
                /*
                 * Beginning of a new field.
                 */
                if (config->ignore_leading_spaces) {
                    while (pos < ts->end && *pos == ' ') {
                        pos++;
                    }
                    if (pos == ts->end) {
                        break;
                    }
                }
                /* Setting chunk effectively starts the field */
                if (*pos == config->quote) {
                    is_quoted = true;
                    ts->state = TOKENIZE_QUOTED;
                    pos++;
                }
                else {
                    is_quoted = false;
                    ts->state = TOKENIZE_UNQUOTED;
                }
                break;

            case TOKENIZE_CHECK_COMMENT:
                if (*pos == config->comment[1]) {
                    ts->state = TOKENIZE_FINALIZE_LINE;
                    pos++;
                }
                else {
                    /* Not a comment, must be tokenizing unquoted now */
                    ts->state = TOKENIZE_UNQUOTED;
                    /* Copy comment as a chunk to the current field */
                    chunk_start = config->comment;
                    chunk_end = chunk_start + 1;
                }
                break;

            case TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE:
                if (*pos == config->quote) {
                    ts->state = TOKENIZE_QUOTED;
                    pos++;
                }
                else {
                    /* continue parsing as if unquoted */
                    ts->state = TOKENIZE_UNQUOTED;
                }
                break;

            case TOKENIZE_FINALIZE_LINE:
                if (ts->buf_state != BUFFER_MAY_CONTAIN_NEWLINE) {
                    ts->state = TOKENIZE_INIT;
                    ts->pos = ts->end;  /* advance to next buffer */
                    goto finish;
                }
                for (; pos < ts->end; pos++) {
                    if (*pos == '\r' || *pos == '\n') {
                        ts->state = TOKENIZE_EAT_CRLF;
                        break;
                    }
                }
                break;

            case TOKENIZE_EAT_CRLF:
                /*
                 * "Universal newline" support any two-character combination
                 * of `\n` and `\r`.
                 * TODO: Should probably only trigger for \r\n?
                 */
                ts->state = TOKENIZE_INIT;
                if (*pos == '\n' || *pos == '\r') {
                    pos++;
                }

                ts->pos = pos;
                goto finish;

            case TOKENIZE_FINALIZE_FILE:
                break;  /* don't do anything,just need to copy back */

            default:
                assert(0);
        }

        /* Copy whatever chunk we currently have */
        if (chunk_start != NULL) {
            if (copy_to_field_buffer(ts,
                    chunk_start, chunk_end, started_saving_word,
                    &word_start, &word_length) < 0) {
                return -1;
            }
            started_saving_word = true;
            chunk_start = NULL;
        }

        if (started_saving_word && (ts->state & TOKENIZE_OUTSIDE_FIELD)) {
            /* A field was fully copied, finalize it */
            if (add_field(ts, word_start, word_length, is_quoted) < 0) {
                return -1;
            }
            started_saving_word = false;
        }
    }

  finish:
    if (ts->num_fields == 1 && ts->fields[0].length == 0) {
        ts->num_fields--;
    }
    return res;
}


void
tokenizer_clear(tokenizer_state *ts)
{
    PyMem_FREE(ts->field_buffer);
    ts->field_buffer = NULL;
    ts->field_buffer_length = 0;

    PyMem_FREE(ts->fields);
    ts->fields = NULL;
    ts->fields_size = 0;
}


void
tokenizer_init(tokenizer_state *ts, parser_config *config)
{
    ts->state = TOKENIZE_INIT;
    ts->num_fields = 0;

    ts->buf_state = 0;
    ts->pos = NULL;
    ts->end = NULL;

    ts->field_buffer = NULL;
    ts->field_buffer_length = 0;

    ts->fields = NULL;
    ts->fields_size = 0;
}
