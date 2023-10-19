"""

This module contains unit tests for scores.stats.tests.acovf

Package: https://www.statsmodels.org/devel/

## Source License

Copyright (C) 2006, Jonathan E. Taylor
All rights reserved.

Copyright (c) 2006-2008 Scipy Developers.
All rights reserved.

Copyright (c) 2009-2018 statsmodels Developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of statsmodels nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import pytest
from numpy.testing import assert_equal

from scores.stats.statistical_tests.acovf import _next_regular


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 8),
        (8, 8),
        (14, 15),
        (15, 15),
        (16, 16),
        (17, 18),
        (1021, 1024),
        (1536, 1536),
        (51200000, 51200000),
        (510183360, 510183360),
        (510183360 + 1, 512000000),
        (511000000, 512000000),
        (854296875, 854296875),
        (854296875 + 1, 859963392),
        (196608000000, 196608000000),
        (196608000000 + 1, 196830000000),
        (8789062500000, 8789062500000),
        (8789062500000 + 1, 8796093022208),
        (206391214080000, 206391214080000),
        (206391214080000 + 1, 206624260800000),
        (470184984576000, 470184984576000),
        (470184984576000 + 1, 470715894135000),
        (7222041363087360, 7222041363087360),
        (7222041363087360 + 1, 7230196133913600),
        # power of 5    5**23
        (11920928955078125, 11920928955078125),
        (11920928955078125 - 1, 11920928955078125),
        # power of 3    3**34
        (16677181699666569, 16677181699666569),
        (16677181699666569 - 1, 16677181699666569),
        # power of 2   2**54
        (18014398509481984, 18014398509481984),
        (18014398509481984 - 1, 18014398509481984),
        # above this), int(ceil(n)) == int(ceil(n+1))
        (19200000000000000, 19200000000000000),
        (19200000000000000 + 1, 19221679687500000),
        (288230376151711744, 288230376151711744),
        (288230376151711744 + 1, 288325195312500000),
        (288325195312500000 - 1, 288325195312500000),
        (288325195312500000, 288325195312500000),
        (288325195312500000 + 1, 288555831593533440),
        # power of 3    3**83
        (3**83 - 1, 3**83),
        (3**83, 3**83),
        # power of 2     2**135
        (2**135 - 1, 2**135),
        (2**135, 2**135),
        # power of 5      5**57
        (5**57 - 1, 5**57),
        (5**57, 5**57),
        # http,//www.drdobbs.com/228700538
        # 2**96 * 3**1 * 5**13
        (2**96 * 3**1 * 5**13 - 1, 2**96 * 3**1 * 5**13),
        (2**96 * 3**1 * 5**13, 2**96 * 3**1 * 5**13),
        (2**96 * 3**1 * 5**13 + 1, 2**43 * 3**11 * 5**29),
        # 2**36 * 3**69 * 5**7
        (2**36 * 3**69 * 5**7 - 1, 2**36 * 3**69 * 5**7),
        (2**36 * 3**69 * 5**7, 2**36 * 3**69 * 5**7),
        (2**36 * 3**69 * 5**7 + 1, 2**90 * 3**32 * 5**9),
        # 2**37 * 3**44 * 5**42
        (2**37 * 3**44 * 5**42 - 1, 2**37 * 3**44 * 5**42),
        (2**37 * 3**44 * 5**42, 2**37 * 3**44 * 5**42),
        (2**37 * 3**44 * 5**42 + 1, 2**20 * 3**106 * 5**7),
    ],
)
def test_next_regular(x, y):
    assert_equal(_next_regular(x), y)
