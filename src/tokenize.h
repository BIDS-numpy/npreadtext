
#ifndef _TOKENIZE_H_
#define _TOKENIZE_H_

#include <Python.h>
#include "stream.h"
#include "parser_config.h"


typedef enum {
    /* Initialization of fields */
    TOKENIZE_INIT,
    TOKENIZE_CHECK_QUOTED,
    /* Main field parsing states */
    TOKENIZE_UNQUOTED,
    TOKENIZE_UNQUOTED_WHITESPACE,
    TOKENIZE_QUOTED,
    /* Handling of two character control sequences (except "\r\n") */
    TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE,
    /* Line end handling */
    TOKENIZE_LINE_END,
    TOKENIZE_EAT_CRLF,  /* "\r\n" support (carriage return, line feed) */
    TOKENIZE_GOTO_LINE_END,
} tokenizer_parsing_state;



typedef struct {
    size_t offset;
    bool quoted;
} field_info;


typedef struct {
    tokenizer_parsing_state state;
    /* Either TOKENIZE_UNQUOTED or TOKENIZE_UNQUOTED_WHITESPACE: */
    tokenizer_parsing_state unquoted_state;
    int unicode_kind;
    int buf_state;
    size_t num_fields;
    /* the buffer we are currently working on */
    char *pos;
    char *end;
    /*
     * Space to copy words into.  The buffer must always be at least two NUL
     * entries longer (8 bytes) than the actual word (including initially).
     * The first byte beyond the current word is always NUL'ed on write, the
     * second byte is there to allow easy appending of an additional empty
     * word at the end (this word is also NUL terminated).
     */
    size_t field_buffer_length;
    size_t field_buffer_pos;
    Py_UCS4 *field_buffer;

    /*
     * Fields, including information about the field being quoted.  This
     * always includes one "additional" empty field.  The length of a field
     * is equal to `fields[i+1].offset - fields[i].offset - 1`.
     *
     * The tokenizer assumes at least one field is allocated.
     */
    field_info *fields;
    size_t fields_size;
} tokenizer_state;


void
tokenizer_clear(tokenizer_state *ts);


int
tokenizer_init(tokenizer_state *ts, parser_config *config);

int
tokenize(stream *s, tokenizer_state *ts, parser_config *const config);

#endif
