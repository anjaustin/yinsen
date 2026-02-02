/*
 * YINSEN CANONICAL TRIT ENCODING
 *
 * Single source of truth for the 2-bit trit encoding used by all backends.
 * All backends (CPU, Metal, NEON, SME) MUST use this encoding.
 * Packed weight buffers are portable across all backends without conversion.
 *
 * Encoding (2 bits per trit):
 *   00 (0) = 0   (zero / skip)
 *   01 (1) = +1  (add)
 *   10 (2) = -1  (subtract)
 *   11 (3) = reserved (decoded as 0, must not be produced by encoders)
 *
 * Rationale: This is the natural unsigned mapping. The value 2 in unsigned
 * binary maps to -1 in the ternary interpretation. This matches the NEON
 * TBL decode table {0, 1, -1, 0} and the SME cmpeq patterns directly.
 */

#ifndef YINSEN_TRIT_ENCODING_H
#define YINSEN_TRIT_ENCODING_H

#define TRIT_ZERO     0x0  /* 00 - zero / skip */
#define TRIT_POS      0x1  /* 01 - add */
#define TRIT_NEG      0x2  /* 10 - subtract */
#define TRIT_RESERVED 0x3  /* 11 - must not be produced by encoders */

#endif /* YINSEN_TRIT_ENCODING_H */
