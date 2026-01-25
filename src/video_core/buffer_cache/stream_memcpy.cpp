/*
 * Copyright © 2013 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Eric Anholt <eric@anholt.net>
 *    Matt Turner <mattst88@gmail.com>
 *
 */

#include <bit>
#include "common/assert.h"
#include "stream_memcpy.h"
#include <cstring>
#include <smmintrin.h>

#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

static inline uintptr_t
align_uintptr(uintptr_t value, uintptr_t alignment)
{
    ASSERT(std::has_single_bit(alignment));
    return ALIGN_POT(value, alignment);
}

void util_streaming_load_memcpy(void* dst, void* src, u64 len)
{
    char* d = (char*)dst;
    char* s = (char*)src;

    /* If dst and src are not co-aligned, or if non-temporal load instructions
     * are not present, fallback to memcpy(). */
    if (((uintptr_t)d & 15) != ((uintptr_t)s & 15)) {
        memcpy(d, s, len);
        return;
    }

    /* memcpy() the misaligned header. At the end of this if block, <d> and <s>
     * are aligned to a 16-byte boundary or <len> == 0.
     */
    if ((uintptr_t)d & 15) {
        uintptr_t bytes_before_alignment_boundary = 16 - ((uintptr_t)d & 15);
        ASSERT(bytes_before_alignment_boundary < 16);

        memcpy(d, s, std::min(bytes_before_alignment_boundary, len));

        d = (char *)align_uintptr((uintptr_t)d, 16);
        s = (char *)align_uintptr((uintptr_t)s, 16);
        len -= std::min(bytes_before_alignment_boundary, len);
    }

    if (len >= 16)
        _mm_mfence();

    while (len >= 16) {
        __m128i* dst_cacheline = (__m128i*)d;
        __m128i* src_cacheline = (__m128i*)s;
        __m128i temp1 = _mm_stream_load_si128(src_cacheline + 0);
        _mm_store_si128(dst_cacheline + 0, temp1);
        d += 16;
        s += 16;
        len -= 16;
    }

    /* memcpy() the tail. */
    if (len) {
        memcpy(d, s, len);
    }
}
