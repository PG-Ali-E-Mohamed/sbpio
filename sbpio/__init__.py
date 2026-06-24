"""
sbpio — Sub-Bottom Profiler I/O and Processing Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read, process, and write sub-bottom profiler (SBP) seismic data
in SEG-Y format.

Quick start::

    from sbpio import sbp

    data = sbp("profile.segy")
    data.show()

    data.write_processed(
        "output.segy",
        correct_delay=True,
        tvg={"method": 3, "kwargs": {"n": 0.2}},
        recoverable_penetration=200,
        out_projection="EPSG:32618",
    )
"""
import warnings
# This suppresses all warnings
warnings.filterwarnings("ignore")

from .core import nav, sbp
from .utils import (
    apply_tvg,
    clear_multiples,
    clear_water,
    decode_nav,
    degrees_minutes_seconds_to_decimal_degrees,
    enhance,
    extract_seafloor,
    find_line_polygon_intersection_indices,
    get_delay_recording_time,
    get_water_depth,
    mfv_fltr,
    non_duplicates,
    plot_sec,
    power_gain_1,
    power_gain_2,
    power_gain_3,
    samples_after_delay,
    scale_to_range,
    tv_gamma,
    utm_to_decimal_degrees,
    water_depth_to_time,
)

__version__ = "0.1.0"
__author__ = "Ali Mohamed"
__license__ = "MIT"

__all__ = [
    # Classes
    "sbp",
    "nav",
    # Visualisation
    "enhance",
    "plot_sec",
    # Navigation
    "decode_nav",
    "degrees_minutes_seconds_to_decimal_degrees",
    "utm_to_decimal_degrees",
    "find_line_polygon_intersection_indices",
    # SEG-Y helpers
    "samples_after_delay",
    "get_delay_recording_time",
    "get_water_depth",
    "non_duplicates",
    # Time / depth
    "water_depth_to_time",
    # Gain functions
    "apply_tvg",
    "tv_gamma",
    "power_gain_1",
    "power_gain_2",
    "power_gain_3",
    # Masking
    "clear_water",
    "clear_multiples",
    # Filters
    "mfv_fltr",
    "scale_to_range",
]
