# sbpio

**Sub-Bottom Profiler I/O and Processing Toolkit**

A Python library for reading, processing, and writing sub-bottom profiler (SBP) data stored in SEG-Y format.

---

## Features

- Load SEG-Y SBP files 
- Apply duplicate-trace removal
- Apply time-varying gain (TVG) with built-in or custom functions
- Recording-delay time correction
- Water-column and multiple muting
- Shot-point navigation extraction, reprojection, and shapefile export
- Cropping in specific time and traces ranges or within ROI polygon
- Coordinate reprojection (e.g., from arc seconds to UTM)
- One-shot `write_processed()` pipeline for batch export

---

## Installation

### From PyPI 

```bash
pip install sbpio
```

### From GitHub

```bash
pip install git+https://github.com/PG-Ali-E-Mohamed/sbpio.git
```

## Quick Start

```python
from sbpio import sbp

# Open a SEG-Y file
data = sbp("profile.segy")

# Display the envelope attribute
data.show()

# One-shot processing pipeline
data.write_processed(
    "output.segy",
    correct_delay=True,
    tvg={"method": 3, "kwargs": {"n": 0.2}},
    recoverable_penetration=200,
    out_projection="EPSG:32618",
)
```

---

## API Overview

### `sbp` class

The main entry point.

| Method | Description |
|---|---|
| `sbp(fname)` | Open a SEG-Y file |
| `.load()` | Load traces and optionally compute envelope |
| `.show()` | Display the profile |
| `.tvg_gain()` | Apply time-varying gain |
| `.correct_delay()` | Correct recording delay |
| `.clean_water_column()` | Mute the water column |
| `.clean_below_seafloor()` | Mute below seafloor + offset |
| `.crop()` | Define a time/trace crop window |
| `.write()` | Write to SEG-Y |
| `.write_processed()` | Full pipeline in one call |

### `nav` class

Accessed via `data.navs`.

| Method | Description |
|---|---|
| `.shot_points()` | Extract navigation from headers |
| `.coords_2_utm()` | Reproject coordinates to UTM |
| `.to_shp(outfile)` | Export to shapefile (line or points) |

### TVG methods

Three built-in gain functions are available via `tvg_gain(method=N)`:

| Key | Function | Formula |
|---|---|---|
| `1` | `power_gain_1` | `(a·t + b)^n` |
| `2` | `power_gain_2` | `t^n` |
| `3` | `power_gain_3` | `(t/dt)^n` |

The default is `tv_gamma` — a time-varying gamma correction that progressively brightens sub-seafloor reflections.

Pass a custom callable with signature `f(t, traces, **kwargs)` for full control.

---

## SEG-Y Conventions

sbpio follows standard SEG-Y Rev 1 trace header fields:

| Field | Byte location |
|---|---|
| Source X | 73 |
| Source Y | 77 |
| SourceGroupScalar | 69 |
| CoordinateUnits | 89 |
| DelayRecordingTime | 109 |
| SourceWaterDepth | 61 |
| EnergySourcePoint | 17 |

---

## Dependencies

| Package | Purpose |
|---|---|
| `segyio` | SEG-Y read/write |
| `numpy` | Array operations |
| `pandas` | Tabular data (navigation) |
| `geopandas` | Shapefile I/O |
| `shapely` | Geometry operations |
| `pyproj` | Coordinate reprojection |
| `matplotlib` | Plotting |
| `scikit-image` | Image enhancement and resize |
| `tqdm` | Progress bars |
| `scipy` | Interpolation/ Envelope calculation |
