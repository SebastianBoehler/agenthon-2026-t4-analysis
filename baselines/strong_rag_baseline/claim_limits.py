"""## Executive summary (read this first)

Emission limits match the current official Track 4 character caps.
Claim length counts Unicode code points after NFC normalization and stripping.
Citation length counts original source code points. Neither limit counts tokens.
See scorer 1c744e1: alignment.CLAIM_TEXT_MAX_CHARS and scoring.FIGURE_SPAN_CAP.
"""

MAX_CLAIM_CHARS = 4000
MAX_CITATION_CHARS = 8000
