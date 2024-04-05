"""
Estimate autocovariances

Barebones reimplementation of the `acovf` from `statsmodels.api.tsa.acovf`, 
for use only with `scores.stats.test.diebold_mariano`

Package: https://www.statsmodels.org/devel/

Code reference: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/stattools.py

Notes:
    All type checking and other features have been removed, as they aren't needed for the 
    `diebold_mariano` function.

Why:
    Reduce dependant packages

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

import numpy as np

# pylint: skip-file

__all__ = ["acovf"]


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            p2 = 2 ** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def acovf(x):
    """
    Estimate autocovariances.

    Args:
        x (array_like):
            Time series data. Must be 1d.

    Returns:
        (np.ndarray):
            The estimated autocovariances.

    References:
        [1] Parzen, E., 1963. On spectral analysis with missing observations
            and amplitude modulation. Sankhya: The Indian Journal of
            Statistics, Series A, pp.383-392.
    """

    xo = x - x.mean()

    n = len(x)

    d = n * np.ones(2 * n - 1)

    nobs = len(xo)
    n = _next_regular(2 * nobs + 1)
    Frf = np.fft.fft(xo, n=n)
    acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1 :]
    acov = acov.real

    return acov
