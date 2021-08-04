//
// stream_file.c
//
// The public functions defined in this file are
//
//     stream *stream_file(FILE *f, int buffer_size)
//     stream *stream_file_from_filename(char *filename, int buffer_size)
//
// The functions accept a C FILE object or a filename, respectively, and
// create a stream that can be used by the text file reader.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <sys/types.h>
#include <unistd.h>

#include "stream.h"

#define DEFAULT_BUFFER_SIZE 16777216


typedef struct _file_buffer {

    /* The file being read. */
    FILE *file;

    /* Size of the file, in bytes. */
    off_t size;

    /* file position when the file_buffer was created. */
    off_t initial_file_pos;

    int32_t line_number;

    /* Boolean: has the end of the file been reached? */
    bool reached_eof;

    /* Offset in the file of the data currently in the buffer. */
    off_t buffer_file_pos;

    /* Position in the buffer of the next character to read. */
    off_t current_buffer_pos;

    /* Actual number of bytes in the current buffer. (Can be less than buffer_size.) */
    off_t last_pos;

    /* Size (in bytes) of the buffer. */
    off_t buffer_size;

    /* Pointer to the buffer. */
    uint8_t *buffer;

} file_buffer;

#define FB(fb)  ((file_buffer *)fb)


static
int32_t fb_line_number(void *fb)
{
    return FB(fb)->line_number;
}

/*
 *  int _fb_load(void *fb)
 *
 *  Get data from the file into the buffer.
 *
 *  Returns 0 on success.
 *  Returns STREAM_ERROR on error.
 */

static
uint32_t _fb_load(void *fb)
{
    uint8_t *buffer = FB(fb)->buffer;

    if (!FB(fb)->reached_eof && (FB(fb)->current_buffer_pos == FB(fb)->last_pos ||
                                 FB(fb)->current_buffer_pos+1 == FB(fb)->last_pos)) {
        size_t num_read;
        /* k will be either 0 or 1. */
        int k = FB(fb)->last_pos - FB(fb)->current_buffer_pos;
        if (k) {
            buffer[0] = buffer[FB(fb)->current_buffer_pos];
        }

        FB(fb)->buffer_file_pos = ftell(FB(fb)->file) - k;
        
        num_read = fread(&(buffer[k]), 1, FB(fb)->buffer_size - k, FB(fb)->file);

        FB(fb)->current_buffer_pos = 0;
        FB(fb)->last_pos = num_read + k;
        if (num_read < FB(fb)->buffer_size - k) {
            if (feof(FB(fb)->file)) {
                FB(fb)->reached_eof = 1;
            }
            else {
                return STREAM_ERROR;
            }
        }
    }
    return 0;
}


/*
 *  char32_t fb_fetch(file_buffer *fb)
 *
 *  Get a single character from the buffer, and advance the buffer pointer.
 *
 *  Returns STREAM_EOF when the end of the file is reached.
 *  The sequence '\r\n' is treated as a single '\n'.  That is, when the next
 *  two bytes in the buffer are '\r\n', the buffer pointer is advanced by 2
 *  and '\n' is returned.
 *  When '\n' is returned, fb->line_number is incremented.
 */

static
char32_t fb_fetch(void *fb)
{
    int32_t status;
    char32_t c;
    uint8_t *buffer = FB(fb)->buffer;
  
    status = _fb_load(fb);
    if (status != 0) {
        return status;
    }

    if (FB(fb)->current_buffer_pos == FB(fb)->last_pos) {
        return STREAM_EOF;
    }

    if ((FB(fb)->current_buffer_pos + 1 < FB(fb)->last_pos) &&
            (buffer[FB(fb)->current_buffer_pos] == '\r') &&
            (buffer[FB(fb)->current_buffer_pos + 1] == '\n')) {
        c = '\n';
        FB(fb)->current_buffer_pos += 2;
    } else {
        c = buffer[FB(fb)->current_buffer_pos];
        FB(fb)->current_buffer_pos += 1;
    }
    if (c == '\n') {
        FB(fb)->line_number++;
    }
    return c;
}


/*
 *  char32_t fb_next(file_buffer *fb)
 *
 *  Returns the next byte in the buffer, but does not advance the pointer.
 *  If the next two characters in the buffer are "\r\n", '\n' is returned.
 */

static
char32_t fb_next(void *fb)
{
    char32_t c;
    uint8_t *buffer = FB(fb)->buffer;

    _fb_load(fb);
 
    if (FB(fb)->current_buffer_pos == FB(fb)->last_pos) {
        return STREAM_EOF;
    }

    if ((FB(fb)->current_buffer_pos + 1 < FB(fb)->last_pos) &&
            (buffer[FB(fb)->current_buffer_pos] == '\r') &&
            (buffer[FB(fb)->current_buffer_pos + 1] == '\n')) {
        c = '\n';
    }
    else {
        c = buffer[FB(fb)->current_buffer_pos];
    }
    return c;
}

/*
 *  fb_skipline(void *fb)
 *
 *  Read bytes from the buffer until a newline or the end of the file is reached.
 *
 *  The return value is 0 if no errors occurred.
 */

static
uint32_t fb_skipline(void *fb)
{
    char32_t c;

    c = fb_next(fb);
    if (c == STREAM_ERROR) {
        return c;
    }
    while (c != '\n' && c != STREAM_EOF) {
        fb_fetch(fb);
        c = fb_next(fb);
        if (c == STREAM_ERROR) {
            return c;
        }
    }
    if (c == '\n') {
        fb_fetch(fb);
    }
    return 0;
}


/*
 *  fb_skiplines(void *fb, int num_lines)
 *
 *  Skip num_lines; calls fb_skipline(fb) num_lines times, or until the end of
 *  the file is reached.
 *
 *  The return value is 0 if no errors occurred.
 */

uint32_t
fb_skiplines(void *fb, int num_lines)
{
    int32_t status;
    char32_t c;

    while (num_lines > 0) {
        status = fb_skipline(fb);
        if (status != 0) {
            return status;
        }
        c = fb_next(fb);
        if (c == STREAM_EOF) {
            break;
        }
        if (c == STREAM_ERROR) {
            // Error
            return c;
        }
        --num_lines;
    }
    return 0;
}

long int
fb_tell(void *fb)
{
    return ftell(FB(fb)->file);
}

int
fb_seek(void *fb, long int pos)
{
    // Not correct, but for now, assume pos == 0.
    FB(fb)->line_number = 1;
    FB(fb)->buffer_file_pos = FB(fb)->initial_file_pos;
    FB(fb)->current_buffer_pos = 0;
    FB(fb)->last_pos = 0;
    FB(fb)->reached_eof = false;
    return fseek(FB(fb)->file, pos, SEEK_SET);
}

static int
stream_del(stream *strm, int restore)
{
    file_buffer *fb = (file_buffer *) (strm->stream_data);

    if (restore == RESTORE_INITIAL) {
        fseek(FB(fb)->file, FB(fb)->initial_file_pos, SEEK_SET);
    }
    else if (restore == RESTORE_FINAL) {
        fseek(FB(fb)->file, FB(fb)->buffer_file_pos + FB(fb)->current_buffer_pos, SEEK_SET);
    }

    free(FB(fb)->buffer);
    free(fb);
    free(strm);

    return 0;
}


/*
 *  stream *stream_file(FILE *f, int buffer_size)
 *
 *  Allocate a new file_buffer.
 *  Returns NULL if the memory allocation fails.
 */

stream *
stream_file(FILE *f, int buffer_size)
{
    file_buffer *fb;
    stream *strm;

    fb = (file_buffer *) malloc(sizeof(file_buffer));
    if (fb == NULL) {
        fprintf(stderr, "stream_file: malloc() failed.\n");
        return NULL;
    }

    strm = (stream *) malloc(sizeof(stream));
    if (strm == NULL) {
        fprintf(stderr, "stream_file: malloc() failed.\n");
        free(fb);
        return NULL;
    }

    fb->file = f;
    fb->initial_file_pos = ftell(f);

    fb->line_number = 1;

    fb->buffer_file_pos = fb->initial_file_pos;

    fb->current_buffer_pos = 0;
    fb->last_pos = 0;

    fb->reached_eof = 0;

    if (buffer_size < 1) {
        buffer_size = DEFAULT_BUFFER_SIZE;
    }

    fb->buffer_size = buffer_size;
    fb->buffer = malloc(fb->buffer_size);
    if (fb->buffer == NULL) {
        // FIXME: don't print to stderr!
        fprintf(stderr, "stream_file: malloc() failed.\n");
        free(fb);
        fb = NULL;
    }

    strm->stream_data = (void *) fb;
    strm->stream_fetch = &fb_fetch;
    strm->stream_peek = &fb_next;
    strm->stream_skipline = &fb_skipline;
    strm->stream_skiplines = &fb_skiplines;
    strm->stream_linenumber = &fb_line_number;
    strm->stream_tell = &fb_tell;
    strm->stream_seek = &fb_seek;
    strm->stream_close = &stream_del;

    return strm;
}


stream *
stream_file_from_filename(char *filename, int buffer_size)
{
    FILE *fp;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        return NULL;
    }

    return stream_file(fp, buffer_size);    
}
