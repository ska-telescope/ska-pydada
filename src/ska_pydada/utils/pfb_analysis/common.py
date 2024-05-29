# -*- coding: utf-8 -*-
#
# This file is part of the SKA PYDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""Common code used for both spectral and temporal fidelity."""

from typing import Union

import numpy as np

NEG_100_DB: float = -100.0
"""Constant representing -100 dB."""

POWER_NEG_100_DB: float = 1e-10  # this is pow(10, NEG_100_DB/10)
"""Constant representing -100 dB but as relative power not in dB."""


def power_as_db(power: Union[np.ndarray, float]) -> np.ndarray:
    """Convert a Numpy array or float from relative power to a decibel (dB) value.

    Note: as this uses ``np.log10`` it will return a Numpy array even
    if the value was a singular float value.

    :param power: a Numpy array or float that is the relative power.
    :type power: Union[np.ndarray, float]
    :return: an array of values that is the power in decibels.
    :rtype: np.ndarray
    """
    return 10.0 * np.log10(power)
