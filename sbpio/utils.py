"""
Utility functions for SBP data I/O and processing.

Covers:
- Signal gain and amplitude processing (TVG, power gain, gamma gain)
- Navigation coordinate decoding and conversion
- SEG-Y header attribute extraction helpers
- Image / section visualisation helpers
- Spatial filtering utilities
"""

from __future__ import annotations

from inspect import isfunction
from typing import Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio as sg
from pyproj import Transformer
from shapely.geometry import Point
from skimage.exposure import rescale_intensity
import math 
from tqdm import tqdm
from scipy.interpolate import interp1d as _interp1d
from scipy.signal import hilbert
from scipy.ndimage import maximum_filter1d

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def envelope(traces):
    """Calculates the envelope attribute
    Parameters
    --------------
    traces: traces with shape (n_sample, n_traces)

    return envelope attribute
    """
    return np.abs(hilbert(traces, axis=0))
    
def enhance(im: np.ndarray, p1: float = 1, p2: float = 99) -> np.ndarray:
    """Enhance an image by clipping at percentiles and rescaling intensity.

    Parameters
    ----------
    im : np.ndarray
        Input image array.
    p1, p2 : float
        Lower/upper percentile bounds for clipping (defaults: 1 and 99).

    Returns
    -------
    np.ndarray
        Intensity-rescaled image.
    """
    mn = np.percentile(im, p1)
    mx = np.percentile(im, p2)
    return rescale_intensity(im, (mn, mx))


def plot_sec(
    sec_: np.ndarray,
    enhance_: bool = True,
    p1: float = 1,
    p2: float = 99,
    cmap: str = "gray_r",
    aspect: str = "auto",
    newfig: bool = True,
    fgsz: Tuple[float, float] = (15, 10),
    alpha: float = 1.0,
    **kwargs,
):
    """Display a seismic section using matplotlib.

    Parameters
    ----------
    sec_ : np.ndarray
        2-D seismic section (samples × traces).
    enhance_ : bool
        Apply intensity enhancement before display.
    p1, p2 : float
        Percentile bounds used by :func:`enhance`.
    cmap : str
        Matplotlib colormap name.
    aspect : str
        Image aspect ratio passed to ``imshow``.
    newfig : bool
        Create a new figure before plotting.
    fgsz : tuple
        Figure size ``(width, height)`` in inches.
    alpha : float
        Image transparency.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    sec = np.nan_to_num(sec_)
    if newfig:
        plt.figure(figsize=fgsz)
    if enhance_:
        im = enhance(sec, p1, p2)
        vmx = abs(im).max()
        return plt.imshow(im, vmin=-vmx, vmax=vmx, cmap=cmap, aspect=aspect, alpha=alpha, **kwargs)
    vmx = abs(sec).max()
    return plt.imshow(sec, vmin=-vmx, vmax=vmx, cmap=cmap, aspect=aspect, alpha=alpha, **kwargs)


# ---------------------------------------------------------------------------
# Navigation / coordinate utilities
# ---------------------------------------------------------------------------

def degrees_minutes_seconds_to_decimal_degrees(val: np.ndarray) -> np.ndarray:
    """Convert a DMS-encoded integer (DDMMSS.ss) to decimal degrees.

    Parameters
    ----------
    val : array-like
        Values encoded as ``DDMMSS`` (e.g. 123456 → 12° 34′ 56″).

    Returns
    -------
    np.ndarray
        Decimal degree values.
    """
    val = np.asarray(val)
    degrees = (val // 10000).astype("int")
    minutes = ((val % 10000) // 100).astype("int")
    seconds = val % 100
    return degrees + minutes / 60 + seconds / 3600


def decode_nav(
    x_: np.ndarray,
    y_: np.ndarray,
    units: Union[int, np.ndarray] = 2,
    scalar: Union[float, np.ndarray] = -1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode and convert SEG-Y navigation coordinates to decimal degrees.

    Parameters
    ----------
    x_ : array-like
        Raw X (longitude / easting) values from the trace header.
    y_ : array-like
        Raw Y (latitude / northing) values from the trace header.
    units : int or array-like
        SEG-Y coordinate units flag:
        1 = length/UTM, 2 = arc-seconds, 3 = decimal degrees, 4 = DMS.
    scalar : float or array-like
        SEG-Y ``SourceGroupScalar`` value(s).

    Returns
    -------
    lon : np.ndarray
        Longitude in decimal degrees (or easting when UTM).
    lat : np.ndarray
        Latitude in decimal degrees (or northing when UTM).
    utm : np.ndarray of bool
        ``True`` where coordinates are in UTM / length units.
    """
    xs, ys = np.array([x_, y_]).astype("float")
    u = np.array(units)
    sc = np.array(scalar).astype("float")

    sc[sc < 0] = 1.0 / np.abs(sc[sc < 0])
    sc[sc == 0] = 1.0

    xs *= sc
    ys *= sc

    utm = np.zeros_like(xs, dtype=bool)
    lon = xs.copy()
    lat = ys.copy()

    utm[u == 1] = True
    lon[u == 2] /= 3600.0
    lat[u == 2] /= 3600.0
    lon[u == 4] = degrees_minutes_seconds_to_decimal_degrees(lon[u == 4])
    lat[u == 4] = degrees_minutes_seconds_to_decimal_degrees(lat[u == 4])

    return lon, lat, utm


def utm_to_decimal_degrees(
    easting: np.ndarray,
    northing: np.ndarray,
    epsg_code: Union[int, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert UTM coordinates to decimal-degree latitude / longitude.

    Parameters
    ----------
    easting : array-like
        UTM easting in metres.
    northing : array-like
        UTM northing in metres.
    epsg_code : int or str
        EPSG code for the source UTM zone (e.g. 32618 for UTM zone 18N).

    Returns
    -------
    latitude, longitude : np.ndarray
        Decimal-degree coordinates.
    """
    transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(easting, northing)
    return latitude, longitude


def find_line_polygon_intersection_indices(
    shapefile_path: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    crs: str = "EPSG:4326",
) -> Tuple[Optional[int], Optional[int]]:
    """Return the first and last indices where a coordinate track crosses a polygon.

    Parameters
    ----------
    shapefile_path : str
        Path to the region-of-interest shapefile.
    x_coords : array-like
        X coordinates (longitude / easting) of the survey line.
    y_coords : array-like
        Y coordinates (latitude / northing) of the survey line.
    crs : str
        CRS of the input coordinates (default: ``'EPSG:4326'``).

    Returns
    -------
    start_idx, end_idx : int or None
        First and last indices inside the polygon, or ``(None, None)`` if
        no intersection is found.
    """
    roi_gdf = gpd.read_file(shapefile_path)
    if roi_gdf.crs is None:
        roi_gdf.set_crs(crs, inplace=True)
    elif roi_gdf.crs != crs:
        roi_gdf = roi_gdf.to_crs(crs)

    roi_polygon = roi_gdf.unary_union
    points = [Point(x, y) for x, y in zip(x_coords, y_coords)]
    inside_mask = [roi_polygon.intersects(pt) for pt in points]
    intersecting_indices = np.where(inside_mask)[0]

    if len(intersecting_indices) == 0:
        print("The line sequence does not intersect the ROI.")
        return None, None

    return int(intersecting_indices[0]), int(intersecting_indices[-1])


# ---------------------------------------------------------------------------
# SEG-Y header helpers
# ---------------------------------------------------------------------------
def get_byte_positions():
    byte_position = {
    'Trace number': 5,
    'Shotpoint number': 17,
    'Number of samples': 115,
    'Coordinate scaling factor': 71,
    'Longitude': 73,
    'Latitude': 77,
    'Coordinate units': 89,
    'Delay recording time': 109,
    'Water depth': 61}
    return byte_position
    
def samples_after_delay(segyio_file, byte_position=None) -> int:
    """Calculate the updated sample count after delay correction.

    Parameters
    ----------
    segyio_file : segyio.SegyFile
        An open segyio file handle.

    Returns
    -------
    int
        New total number of samples.
    """
    byte_position_ = sg.TraceField if byte_position is None else byte_position
    delay = np.array(segyio_file.attributes(byte_position_.DelayRecordingTime)) / 1000
    dt = sg.tools.dt(segyio_file) * 1e-6
    pad = int(np.ceil((delay.max() - delay.min()) / dt))
    return segyio_file.samples.size + pad


def get_delay_recording_time(segyio_fh, byte_position=None) -> np.ndarray:
    """Return the  delay recording time in seconds for every trace.

    Parameters
    ----------
    segyio_fh : segyio.SegyFile
        An open segyio file handle.
    Returns
    -------
    np.ndarray
        Per-trace delay times in seconds.
    """
    byte_position_ = sg.TraceField if byte_position is None else byte_position
    delay = np.array(segyio_fh.attributes(byte_position_.DelayRecordingTime)) / 1000
    return delay


def get_water_depth(segyio_fh, byte_position=None) -> np.ndarray:
    """Return water depth in metres for every trace.

    Parameters
    ----------
    segyio_fh : segyio.SegyFile
        An open segyio file handle.
   

    Returns
    -------
    np.ndarray
        Water depth in metres.
    """
    byte_position_ = sg.TraceField if byte_position is None else byte_position
    wdm = np.array(segyio_fh.attributes(byte_position_.SourceWaterDepth)) / 100
    return wdm


def non_duplicates(segyio_fh, byte_position=None) -> np.ndarray:
    """Return indices of non-duplicate traces (matched on source X/Y).

    Parameters
    ----------
    segyio_fh : segyio.SegyFile
        An open segyio file handle.

    Returns
    -------
    np.ndarray of int
        Trace indices that are **not** duplicates.
    """
    byte_position_ = sg.TraceField if byte_position is None else byte_position
    long = np.array(segyio_fh.attributes(byte_position_.SourceX))
    lat = np.array(segyio_fh.attributes(byte_position_.SourceY))
    df = pd.DataFrame({"x": long, "y": lat})
    df_clear = df.drop_duplicates(inplace=False, ignore_index=False)
    return df_clear.index.values

# ---------------------------------------------------------------------------
# Time / depth conversion
# ---------------------------------------------------------------------------

def water_depth_to_time(water_depth: np.ndarray) -> np.ndarray:
    """Convert water depth (metres) to two-way travel-time (seconds).

    Assumes a water sound velocity of 1520 m/s.

    Parameters
    ----------
    water_depth : array-like
        Water depth in metres.

    Returns
    -------
    np.ndarray
        Two-way travel-time in seconds.
    """
    return np.asarray(water_depth, dtype=float) / 0.7537e3


# ---------------------------------------------------------------------------
# Gain / amplitude correction
# ---------------------------------------------------------------------------

def power_gain_1(
    t: np.ndarray,
    traces: np.ndarray,
    a: float = 200,
    b: float = 1,
    n: float = 2,
) -> np.ndarray:
    """Polynomial TVG: ``gain = (a·t + b)^n``.

    Parameters
    ----------
    t : np.ndarray
        Time in seconds.
    traces : np.ndarray
        Seismic data.
    a, b : float
        Linear coefficients.
    n : float
        Power exponent.
    """
    return traces * (a * t + b) ** n


def power_gain_2(
    t: np.ndarray,
    traces: np.ndarray,
    n: float = 2,
) -> np.ndarray:
    """Power TVG: ``gain = t^n``.

    Parameters
    ----------
    t : np.ndarray
        Time in seconds.
    traces : np.ndarray
        Seismic data.
    n : float
        Power exponent.
    """
    return traces * (t ** n)


def power_gain_3(
    t: np.ndarray,
    traces: np.ndarray,
    dt: Optional[float] = None,
    n: float = 0.2,
) -> np.ndarray:
    """Fractional-power TVG: ``gain = (t/dt)^n``.
    Recommended when seafloor reflections are not blanking deeper ones

    Parameters
    ----------
    t : np.ndarray
        Time in seconds.
    traces : np.ndarray
        Seismic data.
    dt : float, optional
        Sample interval in seconds. Inferred from *t* when not provided.
    n : float
        Power exponent.
    """
    if dt is None:
        if t.ndim == 1:
            tt = t[t > 0]
            dt = float(tt[1] - tt[0])
        else:
            tt = t[:, 0]
            tt = tt[tt > 0]
            dt = float(tt[1] - tt[0])
    return traces * (t / dt) ** n


def scale_to_range(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linearly rescale *x* so its values span ``[a, b]``.

    Parameters
    ----------
    x : array-like
        Input data.
    a, b : float
        Desired minimum and maximum of the output range.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()
    return a + (x - x_min) * (b - a) / (x_max - x_min)


def tv_gamma(
    t: np.ndarray,
    traces: np.ndarray,
    a: float = 0.15,
    b: float = 0.3,
    c: float = 1.2,
) -> np.ndarray:
    """Time-varying gamma gain.

    Applies a depth-dependent gamma correction that diminishes with
    sub-seafloor distance, progressively brightening deeper reflections.

    Parameters
    ----------
    t : np.ndarray
        Time in seconds measured *from* the seafloor (0 = at seafloor).
    traces : np.ndarray
        Seismic amplitude data
    a, b : float
        Lower / upper gamma-intensity bounds.
    c : float
        Luminescence factor (> 1 = brighter).

    Returns
    -------
    np.ndarray
        Gained data.
    """
    tt = 1.0 - np.asarray(t, dtype=float)
    mask = tt != 1
    if mask.any():   # guard: nothing to scale when all samples are at t=0
        tt[mask] = scale_to_range(tt[mask], a, b)
    return traces ** (c * tt)


def apply_tvg(
    traces: np.ndarray,
    t: np.ndarray,
    method: Union[int, callable] = 1,
    **kwargs,
) -> np.ndarray:
    """Apply a TVG (time-varying gain) correction to a seismic section.

    Parameters
    ----------
    traces : np.ndarray
        Seismic data (samples x traces).
    t : np.ndarray
        Time array in seconds (same shape as *traces*).
    method : int or callable
        Built-in options: ``1`` = :func:`power_gain_1`,
        ``2`` = :func:`power_gain_2`, ``3`` = :func:`power_gain_3`.
        A callable accepting keyword arguments ``t`` and ``traces`` may
        also be supplied.

    Returns
    -------
    np.ndarray
        Gained seismic data.
    """
    _funcs = {1: power_gain_1, 2: power_gain_2, 3: power_gain_3}
    try:
        func = _funcs[method]
    except (KeyError, TypeError):
        if isfunction(method):
            func = method
        else:
            raise TypeError(f"Invalid TVG method: {method!r}. Use 1, 2, 3, or a callable.")
    return func(t=t, traces=traces, **kwargs)


# ---------------------------------------------------------------------------
# Section masking
# ---------------------------------------------------------------------------

def clear_water(
    im: np.ndarray,
    sfy: np.ndarray,
    v: float = 0,
) -> np.ndarray:
    """Replace water-column samples (above seafloor) with a constant value.

    Parameters
    ----------
    im : np.ndarray
        2-D section (samples × traces).
    sfy : array-like of int
        Seafloor sample index for each trace.
    v : float
        Replacement value (default 0).

    Returns
    -------
    np.ndarray
    """
    im_ = np.array(im)
    for i, j in enumerate(sfy):
        im_[:j, i] = v
    return im_


def clear_multiples(
    im: np.ndarray,
    sfy: np.ndarray,
    v: float = 0,
) -> np.ndarray:
    """Replace samples below the seafloor index with a constant value.

    Parameters
    ----------
    im : np.ndarray
        2-D section (samples x traces).
    sfy : array-like of int
        Cut-off sample index (seafloor + penetration) for each trace.
    v : float
        Replacement value (default 0).

    Returns
    -------
    np.ndarray
    """
    im_ = np.array(im)
    for i, j in enumerate(sfy):
        im_[j:, i] = v
    return im_


# ---------------------------------------------------------------------------
# MFV (Most Frequent Value) filter
# ---------------------------------------------------------------------------
def mfv(x,tol=0.000001, itermax = 300):
    """
    Calculates Steiner's most frequent value
    Parameters:
        x: 1D array of values
        tol: Tolerance for stopping iteration if the difference between 2 consecutive values = tol
        itermax: Maximum number of iteration steps.
    Returns:
        Most frequent value of x
    """
    m = np.mean(x)
    e = 0.5*np.sqrt(3)*(np.max(x)-np.min(x))
   
    for i in range(1, itermax):
        e_num_sum=np.sum((3*(x - m)**2)/((e**2 + (x - m)**2)**2))
        e_denom_sum=np.sum(1 / (e**2 + (x - m)**2)**2)
        e = np.sqrt(e_num_sum / e_denom_sum)

        m_num_sum = np.sum((e**2 * x) / (e**2 + (x - m)**2))
        m_denom_sum=np.sum(e**2 /(e**2 + (x - m)**2))+0.0001
        m2 = m_num_sum / m_denom_sum
        if math.isnan(m2) is False:
            if abs(m2-m)>tol:
                m = m2
            else:
                break
        else:
            break
    return m

def mfv_fltr(x: np.ndarray, w: int = 5, desc=None) -> np.ndarray:
    """Most-frequent-value sliding-window filter.

    A robust non-linear smoother that replaces each sample with the
    most-frequent value within a symmetric window of half-width *w*.
    Edges are padded with a mirror reflection of the input signal.

    Parameters
    ----------
    x : np.ndarray
        1-D input array.
    w : int
        Filter half-width (window spans ``2w`` samples).
    desc: str: description to appear on the progress bar
    Returns
    -------
    np.ndarray
        Filtered array (same dtype and length as *x*).
    """
    xp = np.zeros(len(x) + 2 * w, dtype=x.dtype)
    xf = np.zeros_like(xp)
    xp[w:-w] = x
    xp[:w] = np.flip(x[:w])
    xp[-w:] = np.flip(x[-w:])
    for i in tqdm(range(w, len(xf) - w), desc, leave=False):
        xf[i] = mfv(xp[i - w: i + w])
    return xf[w:-w]


# ---------------------------------------------------------------------------
# Optional: seafloor detection (requires the ``hydpy`` package)
# ---------------------------------------------------------------------------
def extract_seafloor_(im_, flt_sz=5):
    """
    im: range (-1,1)
    """
    im=im_.copy()
    im[im<0] = 0
    im = im/im.max()
    mask = im**3 >= np.percentile(im**3, 99)
    im = im/im.max(0)
    mask += im**3 >= np.percentile(im**3, 99)
    mask[mask>0]=1
    ss = im*mask
    mxflt = maximum_filter1d(ss, flt_sz, axis=0)
    rs = (ss/mxflt)==1
    return rs

def seafloor_xy(sf, filtered=True, filter_func=mfv_fltr,**kwargs):
    """
    Mask of seafloor extracted by extract_seafloor
    """
    df = pd.DataFrame()
    x,y = np.where(sf.T==1)
    df['x']= x
    df['y'] = y
    df = df.drop_duplicates('x')
    x,y = df.T.values
    if filtered:
        x=np.round(filter_func(x,**kwargs)).astype('int')
        y=np.round(filter_func(y, **kwargs)).astype('int')
    return x,y

def extract_seafloor(im: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect the seafloor horizon in a seismic section.

    .. note::
        Requires the optional ``hydpy`` package (not available on PyPI).
        Place the ``hydpy`` directory on your Python path before calling
        this function.  Also requires ``scipy``.

    Parameters
    ----------
    im : np.ndarray
        2-D seismic section (samples x traces).
    w : int
        Smoothing half-width.

    Returns
    -------
    xx, yy : np.ndarray
        Trace indices and corresponding seafloor sample indices.

    """
  

    sf = extract_seafloor_(im)
    x, y = seafloor_xy(sf, w=w)
    fx = _interp1d(
        x[~np.isnan(y)], y[~np.isnan(y)],
        kind="nearest", fill_value="extrapolate"
    )
    xx = np.arange(im.shape[1])
    yy = fx(xx).astype("int")
    return xx, yy
