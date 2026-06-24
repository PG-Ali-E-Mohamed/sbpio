"""
Core classes for SBP data I/O and processing.

Classes
-------
sbp
    Main class: read, process, and write sub-bottom profiler SEG-Y files.
nav
    Navigation helper: extract, convert, and export shot-point coordinates.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import segyio as sg
from shapely.geometry import LineString
from skimage.transform import resize
from tqdm import tqdm


from .utils import (
    apply_tvg,
    clear_multiples,
    clear_water,
    decode_nav,
    find_line_polygon_intersection_indices,
    get_delay_recording_time,
    get_water_depth,
    non_duplicates,
    plot_sec,
    tv_gamma,
    utm_to_decimal_degrees,
    water_depth_to_time,
    mfv_fltr,
    envelope,
    get_byte_positions
)


# ---------------------------------------------------------------------------
# sbp — main data class
# ---------------------------------------------------------------------------

class sbp:
    """Sub-bottom profiler data container and processor.

    Reads a SEG-Y file, exposes raw traces and their instantaneous-amplitude
    envelope, and provides methods to:

    * apply time-varying gain (TVG),
    * correct recording delay,
    * mask the water column / multiples,
    * crop to a time/trace or polygon-defined window,
    * reproject coordinates,
    * write the result back to SEG-Y.

    Parameters
    ----------
    fname : str
        Path to the input SEG-Y file.
    update_envelope : bool
        Recompute the envelope after each processing step (default ``True``).
    remove_duplicates : bool
        Drop duplicate traces (default ``True``).
    byte_position: dict
        Byte position dictionary. It must follow "spbio.utils.get_byte_positions". 
        sbpio default byte positions are:
                Trace number: 5
                Shotpoint number: 17
                Number of samples: 115
                Coordinate scaling factor: 71
                Longitude: 73
                Latitude: 77
                Coordinate units: 89
                Delay recording time: 109
                Water depth: 61
            
    Examples
    --------
    Quick look::

        from sbpio import sbp
        data = sbp("profile.segy")
        data.show()

    Full one-shot pipeline::

        data = sbp("profile.segy")
        data.write_processed(
            "output.segy",
            correct_delay=True,
            tvg={"method": 3, "kwargs": {"n": 0.2}},
            recoverable_penetration=200,
            out_projection="EPSG:32618",
        )

    To modify byte positions (e.g., coordinates)::
        byte_position_dct = spbio.utils.get_byte_positions
        byte_position_dct['Longitude'] = 181
        byte_position_dct['Latitude'] = 185

        data = sbp("profile.segy", byte_position_dct)

    """

    def __init__(
        self,
        fname: str,
        update_envelope: bool = True,
        remove_duplicates: bool = True,
        byte_position: dict = None
    ) -> None:
        
        self.reader = sg.open(fname, mode="r", strict=False, ignore_geometry=False)
        self.traces: Optional[np.ndarray] = None
        self.envelope: Optional[np.ndarray] = None
        
        rdr = self.reader
        self.trace_header = self.reader.header
        twt = rdr.samples
        n_traces = rdr.tracecount
        
        byte_position = get_byte_positions() if byte_position is None else byte_position
        
        attrs_dct = {
            'Trace number': 'TRACE_SEQUENCE_FILE',
            'Shotpoint number': 'EnergySourcePoint',
            'Number of samples': 'TRACE_SAMPLE_COUNT',
            'Latitude': 'SourceY',
            'Longitude': 'SourceX',
            'Coordinate scaling factor': 'SourceGroupScalar',
            'Coordinate units': 'CoordinateUnits',
            'Delay recording time': 'DelayRecordingTime',
            'Water depth': 'SourceWaterDepth'}
        byte_position_dct = sg.tracefield.keys
        for k in attrs_dct.keys():
            byte_position_dct[attrs_dct[k]] = byte_position[k]
    
        self.byte_position = SimpleNamespace(**byte_position_dct)
        
        self.navs = nav(self)
        
        self.meta: dict = {
            "fname": fname,
            "basename": os.path.basename(fname),
            "update_envelope": update_envelope,
            "remove_duplicates":remove_duplicates,
            "dt": sg.tools.dt(rdr) * 1e-6,       # sample interval in seconds
            "twt": twt,                             # original TWT (ms)
            "twt_new": twt,                         # updated TWT after delay correction
            "n_traces": n_traces,
            "n_samples": twt.size,	
            "n_samples_new": twt.size,
            "good_traces": (
                non_duplicates(rdr,self.byte_position) if remove_duplicates else np.arange(n_traces)
            ),
            "delay": None,
            "delay_original": None,
            "delay_corrected": False,
            "sub_sf": None,                         # samples subsampling factor
            "sub_trf": None,                        # traces subsampling factor
            "water_depth_s": None,
            "crp_smpl_1": 0,
            "crp_smpl_2": None,
            "flat_seafloor":False,
            'smoothed_seafloor_s':None,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self, envelope_: Optional[bool] = None) -> None:
        """Load raw traces from disk into memory.

        Parameters
        ----------
        envelope_ : bool, optional
            Override the ``update_envelope`` flag set at construction.
        """
        print('Loading data...')
        self.traces = sg.tools.collect(self.reader.trace[:]).T

        if envelope_ is None:
            envelope_ = self.meta["update_envelope"]

        if envelope_:
            self.envelope = envelope(self.traces)
            #self.no_gain_envelope = self.envelope.copy() 
            self.subsample_env()
        else:
            self.envelope = None
            self.sub = None
            self.sub_xf = None

    def subsample_env(self) -> None:
        """Create a (600 × 1000) downsampled envelope for rapid display."""
        self.sub = resize(self.envelope, (600, 1000))
        self.meta["sub_sf"] = 600 / self.meta["n_samples_new"]
        self.meta["sub_trf"] = 1000 / self.meta["n_traces"]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def correct_delay(self) -> None:
        """Shift every trace to account for its recording delay.

        After correction all traces share the same (minimum) start time,
        and the section can be interpreted in absolute two-way travel-time.
        Calling this method more than once is a no-op.
        """
        if self.meta["delay_corrected"]:
            return

        ns0 = self.meta["n_samples"]
        ns1 = self.meta["n_traces"]
        dt = self.meta["dt"]

        # Use cached delay if available
        delay = (
            get_delay_recording_time(self.reader, self.byte_position)
            if self.meta["delay"] is None
            else self.meta["delay"]
        )
        self.meta["delay_original"] = delay
        
        delta_delay = np.round((delay - delay.min()) / dt).astype("int")
        pad = int(np.ceil((delay.max() - delay.min()) / dt))
        n_s = ns0 + pad

        if self.traces is None:
            self.load(False)

        cor = np.zeros((n_s, ns1))
        cor[:ns0] = self.traces
        self.meta["delay"] = delay.min() * np.ones(ns1)
        
        #for i, j in enumerate(tqdm(delta_delay, desc="Correcting delay", leave=False)):
        #    cor[:, i] = np.roll(cor[:, i], j)
        self.traces = np.array([np.roll(cor[:, i], j) for i, j in enumerate(tqdm(delta_delay, desc="Correcting delay", leave=False))]).T

        self.meta["n_samples_new"] = n_s
        self.meta["twt_new"] = self.meta["twt"][0] + np.arange(n_s) * dt * 1000
        self.meta["crp_smpl_2"] = n_s
        self.meta["delay_corrected"] = True

        if self.meta["update_envelope"]:
            if self.envelope is None:
                print("Updating envelope…")
                self.envelope = envelope(self.traces)
            else:
                env = np.zeros_like(cor)
                env[:ns0] = self.envelope.copy()
                
                #for i, j in enumerate(tqdm(delta_delay, desc="Updating envelope", leave=False)):
                #    env[:, i] = np.roll(env[:, i], j)

                self.envelope  = np.array([np.roll(env[:, i], j) for i, j in enumerate(tqdm(delta_delay, desc="Updating envelope", leave=False))]).T
                #self.no_gain_envelope  = np.array([np.roll(no_gain_envelope[:, i], j) 
                #                                   for i, j in enumerate(tqdm(delta_delay, desc="Updating no-gain envelope", leave=False))]).T

                idx = (np.arange(1000)/self.meta['sub_trf']).astype('int')
                pad2 = int(pad*self.meta['sub_sf'])
                self.sub = np.pad(self.sub, ((0,pad2),(0,0)))
                self.sub  = np.array([np.roll(self.sub[:, i], j) for i, j in enumerate(tqdm((delta_delay[idx]*self.meta['sub_sf']).astype('int'),
                                                                                            desc="'Updating subsampled envelope'", leave=False))]).T
            #self.subsample_env()

    def tvg_gain(
        self,
        method=tv_gamma,
        offset: float = 0,
        recoverable_penetration: float = 200,
        realize: bool = True,
    ) -> None:
        """Apply time-varying gain (TVG) to the loaded traces.

        Parameters
        ----------
        method : int, dict, or callable
            TVG function specification:

            * ``int`` (1, 2, or 3) — selects a built-in power-gain function.
            * ``dict`` — ``{"method": <int|callable>, "kwargs": {...}}``.
            * ``callable`` — any function with signature
              ``f(t, traces, **kwargs)``.
            * Default is :func:`~sbpio.utils.tv_gamma`.
        offset : float
            Time (ms) above the seafloor at which TVG application starts.
        recoverable_penetration : float
            Maximum sub-seafloor depth (ms) to gain; deeper samples are muted.
        realize : bool
            If ``True`` (default) the data is overwritten with the new gained traces. To test multiple approaches, set realize=False """

        dt = self.meta["dt"]
        s0 = self.meta["n_samples_new"]
        s1 = self.meta["n_traces"]
        # Build time axis (seconds) referenced to the seafloor
        t_s = (np.arange(s0) * dt)[:, np.newaxis] * np.ones((s0, s1))
        
        if self.meta['smoothed_seafloor_s'] is None:
            delay_s = (
                get_delay_recording_time(self.reader,self.byte_position)
                if self.meta["delay"] is None
                else self.meta["delay"]
            )
            self.meta["delay"] = delay_s
            wd_s = (
                water_depth_to_time(get_water_depth(self.reader,self.byte_position))
                if self.meta["water_depth_s"] is None
                else self.meta["water_depth_s"]
            )
            self.meta["water_depth_s"] = wd_s
            
            smoothed_sf = mfv_fltr(wd_s-delay_s, 15,'Smoothing bathymetry')
            self.meta['smoothed_seafloor_s'] = smoothed_sf
        else:
            smoothed_sf = self.meta['smoothed_seafloor_s']
        t_s =  t_s - smoothed_sf + (offset / 1000)
        t_s[t_s < 0] = 0                                # mute water column
        t_s[t_s > recoverable_penetration / 1000] = 0   # mute below penetration

        if self.traces is None:
            self.load(False)

        if isinstance(method, dict):
            mthd = method["method"]
            kwargs = method.get("kwargs", {})
            #gained = apply_tvg(self.traces, t_s, method=mthd, **kwargs)
            
        else:
            mthd = method
            kwargs = {}
            
            #gained = apply_tvg(self.traces, t_s, method=method)
        gained = apply_tvg(self.traces, t_s, method=mthd, **kwargs)

        gained = np.nan_to_num(gained)

        if realize:
            self.traces = gained
        else:
            self.gained = gained

        if self.meta["update_envelope"]:
            print("Calculating envelope…")
            self.envelope = envelope(gained)
            self.subsample_env()

    def clean_water_column(
        self,
        offset: float = 0,
        replace_with: float = 0,
        realize: bool = False,
    ) -> None:
        """Replace water-column samples with a constant value.

        Parameters
        ----------
        offset : float
            Time offset (ms); positive values shift the mask shallower.
        replace_with : float
            Fill value (default 0).
        realize : bool
            Also apply the mask to ``self.traces`` (default: envelope only).
        """
        if self.meta['smoothed_seafloor_s'] is None:
            wd_ts = (
                water_depth_to_time(get_water_depth(self.reader,self.byte_position))
                if self.meta["water_depth_s"] is None
                else self.meta["water_depth_s"]
            )
            
            self.meta["water_depth_s"] = wd_ts
            delay = (
                self.meta["delay"]
                if self.meta["delay"] is not None
                else get_delay_recording_time(self.reader,self.byte_position)
            )
            self.meta["delay"] = delay
            smoothed_sf = mfv_fltr(wd_ts-delay, 15, 'Smoothing bathymetry')
            self.meta['smoothed_seafloor_s']=smoothed_sf
        else:
            smoothed_sf = self.meta['smoothed_seafloor_s']
            
        y_wd = np.round((smoothed_sf - offset / 1000) / self.meta["dt"]).astype("int")

        if self.meta["update_envelope"] and self.envelope is not None:
            self.envelope = clear_water(self.envelope, y_wd, replace_with)
            idx = (np.arange(1000)/self.meta['sub_trf']).astype('int')
            self.sub = clear_water(self.sub, (y_wd[idx]*self.meta['sub_sf']).astype('int'), replace_with)
        if realize and self.traces is not None:
            self.traces = clear_water(self.traces, y_wd, replace_with)

    def clean_below_seafloor(
        self,
        offset: float = -300,
        replace_with: float = 0,
        realize: bool = False,
    ) -> None:
        """Replace samples below the seafloor + offset with a constant value.

        Parameters
        ----------
        offset : float
            Offset (ms) below the seafloor. Default ``-300`` mutes everything
            deeper than 300 ms sub-seafloor (including multiples).
        replace_with : float
            Fill value (default 0).
        realize : bool
            Also apply the mask to ``self.traces`` (default: envelope only).
        """
        if self.meta['smoothed_seafloor_s'] is None:
            wd_ts = (
                water_depth_to_time(get_water_depth(self.reader,self.byte_position))
                if self.meta["water_depth_s"] is None
                else self.meta["water_depth_s"]
            )
            
            self.meta["water_depth_s"] = wd_ts
            delay = (
                self.meta["delay"]
                if self.meta["delay"] is not None
                else get_delay_recording_time(self.reader,self.byte_position)
            )
            self.meta["delay"] = delay
            smoothed_sf = mfv_fltr(wd_ts-delay, 15, 'Smoothing bathymetry')
            self.meta['smoothed_seafloor_s']=smoothed_sf
        else:
            smoothed_sf = self.meta['smoothed_seafloor_s']
        y_wd = np.round((smoothed_sf - offset / 1000) / self.meta["dt"]).astype("int")

        if self.meta["update_envelope"] and self.envelope is not None:
            self.envelope = clear_multiples(self.envelope, y_wd, replace_with)
            idx = (np.arange(1000)/self.meta['sub_trf']).astype('int')
            self.sub = clear_multiples(self.sub, (y_wd[idx]*self.meta['sub_sf']).astype('int'), replace_with)
            
        if realize and self.traces is not None:
            self.traces = clear_multiples(self.traces, y_wd, replace_with)

    def crop(
        self,
        t0: float = 0,
        t1: float = -1,
        tr0: int = 0,
        tr1: int = -1,
        roi_polygon: Optional[str] = None,
        utm_crs: Optional[str] = None,
    ) -> None:
        """Define the ROI time/trace window exported when calling `write`.
            Use reset_roi to reverse this effect
        Parameters
        ----------
        t0, t1 : float
            Start/end two-way travel-time in milliseconds.
        tr0, tr1 : int
            First/last trace numbers.
        roi_polygon : str, optional
            Path to a polygon shapefile.  When given, the crop is clipped to
            the part of the line that falls inside the polygon.
        utm_crs : str, optional
            EPSG code required when navigation is in UTM.
        """
        st_time = self.meta["twt"][0]
        dt = self.meta["dt"]
        n_s = self.meta["n_samples_new"]
        nd_time = st_time + (n_s - 1) * dt * 1000

        t0 = float(st_time) if t0 <= 0 else min(t0, nd_time - dt * 1e-4)
        t1 = float(nd_time) if (t1 == -1 or t1 >= nd_time) else max(t1, t0 + dt * 1e4)

        t0_smpl = max(int((t0 - st_time) / 1000 / dt), 0)
        t1_smpl = min(int((t1 - st_time) / 1000 / dt), int(n_s))

        if roi_polygon is None:
            tr0 = int(max(tr0, 0))
            tr1 = self.meta["n_traces"] if tr1 == -1 else int(tr1)
        else:
            if self.navs.shotpoints is None:
                self.navs.shot_points()
            df = self.navs.shotpoints
            sc=1
            lon, lat, m = decode_nav(df["Long"].values, df["Lat"].values, self.navs.units, sc)
            src = "EPSG:4326"
            
            if m.all():
                if utm_crs is None:
                    raise ValueError("utm_crs must be provided when coordinates are in UTM.")
                else:
                    src=utm_crs
                    sc = np.array(self.navs.scalar).astype("float")
                    sc[sc < 0] = 1.0 / np.abs(sc[sc < 0])
                    sc[sc == 0] = 1.0
    
            tr0, tr1 = find_line_polygon_intersection_indices(roi_polygon, lon*sc, lat*sc,src)

        #self.cropped = self.traces[t0_smpl:t1_smpl, tr0:tr1]
        self.meta["good_traces"] = np.arange(tr0, tr1, dtype="int")
        self.meta["crp_smpl_1"] = t0_smpl
        self.meta["crp_smpl_2"] = t1_smpl
    def reset_roi(self):
        """Resets the ROI boundary to the full profile extent. 
        Use if need to reverse/reset the effect of the crop function"""
        self.meta["good_traces"] = (non_duplicates(self.reader,self.byte_position) 
                                    if self.meta['remove_duplicates'] 
                                    else np.arange(self.meta['n_traces'], dtype='int'))
        self.meta["crp_smpl_1"] = 0
        self.meta["crp_smpl_2"] = self.meta['n_samples_new']
    
    def flatten(self, realize=False, recalculate_delay=False):
        """Flattens the seafloor 
        Parameters
        ----------
        realize: default '''False'''
                If set to True, the full waveform traces are flattened 
        recalculate_delay: default '''False'''. It recalculates delay recording time
        
        """
        if not self.meta['flat_seafloor']:
            if self.envelope is None:
                self.load()
            if self.meta['smoothed_seafloor_s'] is None:
                wd_ts = (
                    water_depth_to_time(get_water_depth(self.reader,self.byte_position))
                    if self.meta["water_depth_s"] is None
                    else self.meta["water_depth_s"]
                )
                
                self.meta["water_depth_s"] = wd_ts
                delay = (
                    self.meta["delay"]
                    if self.meta["delay"] is not None
                    else get_delay_recording_time(self.reader,self.byte_position)
                )
                self.meta["delay"] = delay
                smoothed_sf = mfv_fltr(wd_ts-delay, 15, 'Smoothing bathymetry')
                self.meta['smoothed_seafloor_s']=smoothed_sf
            else:
                smoothed_sf = self.meta['smoothed_seafloor_s']
            dt=self.meta['dt']
            sf= smoothed_sf//dt
            
            self.envelope = np.array([np.roll(self.envelope[:, i], -j) 
                                      for i,j in enumerate(tqdm(sf.astype('int'), 
                                                                'Flattening envelope', leave=False))]).T
            
            idx = (np.arange(1000)/self.meta['sub_trf']).astype('int')
            self.sub = np.array([np.roll(self.sub[:, i], -j) 
                                 for i,j in enumerate(tqdm((sf[idx]*self.meta['sub_sf']).astype('int'),
                                                           'Updating subsampled envelope', leave=False))]).T
            if realize:
                self.traces = np.array([np.roll(self.traces[:, i], -j) for i,j in enumerate(tqdm(sf.astype('int'), 'Flattening Traces', leave=False))]).T
            if recalculate_delay:
                self.meta["delay"] = dl + sf*dt
            self.meta['flat_seafloor'] = True
            
    def deflatten(self, realize=False, recalculate_delay=False):
        """De-flatten the seafloor and places seafloor at the correct relative depth
        Parameters
        ----------
        realize: default '''False'''
                If set to True, the full waveform traces are de-flattened 
        recalculate_delay: default '''False'''. It recalculates delay recording time
                            if was not used while flattening, do not use in deflattening.
        """
        if self.meta['flat_seafloor']:
            if self.envelope is None:
                self.load()
                
            if self.meta['smoothed_seafloor_s'] is None:
                wd_ts = (
                    water_depth_to_time(get_water_depth(self.reader,self.byte_position))
                    if self.meta["water_depth_s"] is None
                    else self.meta["water_depth_s"]
                )
                
                self.meta["water_depth_s"] = wd_ts
                delay = (
                    self.meta["delay"]
                    if self.meta["delay"] is not None
                    else get_delay_recording_time(self.reader,self.byte_position)
                )
                self.meta["delay"] = delay
                smoothed_sf = mfv_fltr(wd_ts-delay, 15, 'Smoothing bathymetry')
                self.meta['smoothed_seafloor_s']=smoothed_sf
            else:
                smoothed_sf = self.meta['smoothed_seafloor_s']
            dt=self.meta['dt']
            sf= smoothed_sf//dt
    
            self.envelope = np.array([np.roll(self.envelope[:, i], j) for i,j in enumerate(tqdm(sf.astype('int'), 'De-flattening envelope', leave=False))]).T
            idx = (np.arange(1000)/self.meta['sub_trf']).astype('int')
            self.sub = np.array([np.roll(self.sub[:, i], j) 
                                 for i,j in enumerate(tqdm((sf[idx]*self.meta['sub_sf']).astype('int'), 
                                                           'Updating subsampled envelope', leave=False))]).T
            if realize:
                self.traces = np.array([np.roll(self.traces[:, i], j) for i,j in enumerate(tqdm(sf.astype('int'), 'De-flattening Traces', leave=False))]).T
            if recalculate_delay:
                self.meta["delay"] = dl - sf*dt
                
            self.meta['flat_seafloor'] = False
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write(
        self,
        output_file: str,
        attr: Optional[np.ndarray] = None,
    ) -> None:
        """Write processed data to a new SEG-Y file.

        Duplicate traces are dropped, the crop window is applied, and
        recording-delay headers are updated to reflect any correction.

        Parameters
        ----------
        output_file : str
            Output SEG-Y path.
        attr : np.ndarray, optional
            Alternative data to write (e.g. an attribute such as envelope).
            The function exports the full waveform by default.
        """
        src = self.reader
        keep_indices = self.meta["good_traces"].astype("int")
        smp1 = self.meta["crp_smpl_1"]
        smp2 = self.meta["crp_smpl_2"]

        spec = sg.tools.metadata(src)
        spec.tracecount = keep_indices.size
        spec.samples = self.meta["twt_new"][smp1:smp2]
        ns = spec.samples.size

        delay = (
            self.meta["delay"]
            if self.meta["delay"] is not None
            else get_delay_recording_time(self.reader,self.byte_position)
        )
        if smp1>0: 
            delay += smp1*self.meta['dt']
            
        delay_ms = (1000 * delay).astype("int")

        traces = self.traces if attr is None else attr
        update_coords = self.navs.coords_converted

        if update_coords:
            shp = self.navs.shotpoints
            lon_out = shp["Long"].values
            lat_out = shp["Lat"].values
            scalar_out = self.navs.scalar.astype("int")
            units_out = self.navs.units.astype("int")

        with sg.create(output_file, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.bin[sg.BinField.Samples] = ns

            for new_idx, old_idx in enumerate(keep_indices):
                dst.trace[new_idx] = traces[smp1:smp2, old_idx]
                dst.header[new_idx] = src.header[old_idx]
                dst.header[new_idx][self.byte_position.TRACE_SAMPLE_COUNT] = ns
                dst.header[new_idx][self.byte_position.TRACE_SEQUENCE_FILE] = new_idx + 1
                dst.header[new_idx][self.byte_position.DelayRecordingTime] = delay_ms[old_idx]

                if update_coords:
                    dst.header[new_idx][self.byte_position.SourceX] = lon_out[old_idx]
                    dst.header[new_idx][self.byte_position.SourceY] = lat_out[old_idx]
                    dst.header[new_idx][self.byte_position.SourceGroupScalar] = scalar_out[old_idx]
                    dst.header[new_idx][self.byte_position.CoordinateUnits] = units_out[old_idx]

    def write_processed(
        self,
        output_file: str,
        correct_delay: bool = True,
        tvg=tv_gamma,
        offset: float = 0,
        recoverable_penetration: float = 200,
        src_projection: str = "EPSG:4326",
        out_projection: Optional[str] = None,
    ) -> None:
        """One-shot processing and export pipeline.

        Applies all standard processing steps in order and writes the
        result to a new SEG-Y file:

        1. Load raw traces.
        2. Remove duplicate traces (handled at construction).
        3. Apply TVG gain.
        4. Apply delay correction.
        5. Reproject coordinates (optional).
        6. Write SEG-Y.

        Parameters
        ----------
        output_file : str
            Output SEG-Y path.
        correct_delay : bool
            Apply recording-delay correction.
        tvg : int, dict, callable, or False
            TVG specification.
            Pass ``False`` to skip gain entirely.
        offset : float
            Time above seafloor (ms) from which TVG is applied.
        recoverable_penetration : float
            Maximum sub-seafloor time (ms) to gain.
        src_projection : str
            EPSG code of the source coordinates (default ``'EPSG:4326'``).
        out_projection : str, optional
            Target EPSG code. Coordinates are reprojected when provided.

        Examples
        --------
        ::

            data = sbp("profile.segy")
            data.write_processed(
                "output.segy",
                tvg={"method": 3, "kwargs": {"n": 0.2}},
                out_projection="EPSG:32618",
            )
        """
        self.meta["update_envelope"] = False
        self.load(False)

        if tvg is not False:
            self.tvg_gain(tvg, offset, recoverable_penetration)

        if correct_delay:
            self.correct_delay()

        if out_projection is not None:
            self.coords_2_utm(src_projection, out_projection)

        self.write(output_file)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def show(
        self,
        enhance_: bool = True,
        p1: float = 1,
        p2: float = 99,
        cmap: str = "gray_r",
        aspect: str = "auto",
        newfig: bool = True,
        fgsz: Tuple[float, float] = (10, 6),
        sub: bool = True,
        t0: float = 0,
        t1: float = -1,
        tr0: int = 0,
        tr1: int = -1,
        seafloor: bool = False,
        seafloor_offset: float = 0,
        roi_bounds: bool = False,
        func=None,
        **kwargs,
    ) -> None:
        """Display the seismic envelope section.

        Loads the data automatically if not already in memory.

        Parameters
        ----------
        enhance_, p1, p2, cmap, aspect, newfig, fgsz
            Passed to :func:`~sbpio.utils.plot_sec`.
        sub : bool
            Show the fast downsampled envelope (True) or full resolution.
        t0, t1 : float
            Display window in milliseconds.
        tr0, tr1 : int
            Display trace range.
        seafloor : bool
            Overlay the water-depth horizon in red.
        seafloor_offset : float
            Shift applied to the seafloor overlay (ms).
        roi_bounds : bool
            Use the current crop ROI limits instead of t0/t1/tr0/tr1.
        func : callable, optional
            Apply this transform to the display array before plotting.
        """
        if self.envelope is None:
            self.load(True)

        if sub:
            if not hasattr(self, "sub") or self.sub is None:
                self.subsample_env()
            to_show = self.sub
            yf = self.meta["sub_sf"]
            xf = self.meta["sub_trf"]
        else:
            to_show = self.envelope
            yf = xf = 1

        st_time = float(self.meta["twt"][0])
        dt = self.meta["dt"]
        n_s = self.meta["n_samples_new"]
        nd_time = st_time + (n_s - 1) * dt * 1000

        if roi_bounds:
            t0_smpl = self.meta["crp_smpl_1"]
            t1_smpl = self.meta["crp_smpl_2"]
            t0 = t0_smpl * dt * 1000 + st_time
            t1 = t1_smpl * dt * 1000 + st_time
            t0_smpl = int(t0_smpl * yf)
            t1_smpl = int(t1_smpl * yf)
            tr0 = int(self.meta["good_traces"][0])
            tr1 = int(self.meta["good_traces"][-1])
        else:
            t0 = st_time if t0 <= 0 else min(t0, nd_time - dt * 1e4)
            t1 = nd_time if (t1 == -1 or t1 >= nd_time) else max(t1, t0 + dt * 1e4)
            t0_smpl = int(yf * (t0 - st_time) / 1000 / dt)
            t1_smpl = int(yf * (t1 - st_time) / 1000 / dt)
            tr0 = max(tr0, 0)
            tr1 = self.meta["n_traces"] if tr1 == -1 else tr1

        tr0_ = int(np.round(tr0 * xf))
        tr1_ = int(np.round(tr1 * xf))

        im = to_show[t0_smpl:t1_smpl, tr0_:tr1_]
        if func is not None:
            im = func(im)

        plot_sec(im, enhance_, p1, p2, cmap, aspect, newfig, fgsz, **kwargs)

        if seafloor:
            if self.meta['smoothed_seafloor_s'] is None:
                wd_ts = (
                    water_depth_to_time(get_water_depth(self.reader, self.byte_position))
                    if self.meta["water_depth_s"] is None
                    else self.meta["water_depth_s"]
                )
                self.meta["water_depth_s"] = wd_ts
                delay = (
                    self.meta["delay"]
                    if self.meta["delay"] is not None
                    else get_delay_recording_time(self.reader,self.byte_position)
                ) 
                smoothed_sf = mfv_fltr(wd_ts - delay, 15, 'Smoothing bathymetry')
                self.meta['smoothed_seafloor_s'] = smoothed_sf
            else:
                smoothed_sf = self.meta['smoothed_seafloor_s']
                
            y_wd = (smoothed_sf - seafloor_offset / 1000) / self.meta["dt"]
            x_wd = np.arange(self.meta["n_traces"])
            plt.plot(x_wd[: tr1 - tr0] * xf, y_wd[tr0:tr1] * yf - t0_smpl, "r")

        xtk0 = np.linspace(0, tr1_ - tr0_, 9, dtype="int")
        xtk1 = np.linspace(tr0, tr1, 9, dtype="int")
        plt.xticks(xtk0, xtk1)
        plt.xlabel("Trace number")

        ytk0 = np.linspace(0, t1_smpl - t0_smpl, 7)
        ytk1 = np.round(np.linspace(t0, t1, 7), 1)
        plt.yticks(ytk0, ytk1)
        plt.ylabel("TWT (ms)")


# ---------------------------------------------------------------------------
# nav — navigation helper
# ---------------------------------------------------------------------------

class nav:
    """Navigation helper attached to an :class:`sbp` instance.

    Extracts source coordinates from SEG-Y trace headers, converts them
    between CRS systems, and exports them as line or point shapefiles.

    Parameters
    ----------
    parent : sbp
        The owning :class:`sbp` instance.
    """

    def __init__(self, parent: sbp) -> None:
        self.parent = parent
        self.reader = parent.reader
        self.shotpoints: Optional[pd.DataFrame] = None
        self.coords_converted: bool = False
        self.coords_prj = None
        r = parent.reader
        scalar = r.header[0][self.parent.byte_position.SourceGroupScalar]
        if scalar==0:
            scalar=1 
        elif scalar<0:
            scalar = 1/abs(scalar)
            
        self.scalar = scalar
        units = r.header[0][self.parent.byte_position.CoordinateUnits]
        
        if units==0:
            print("""Coordinate units are not defined. 
            We're setting it to Arc seconds.
            If this is not correct, you can overwrite units 
            by passing the correct units to .navs.units directly""")
            units=2
            
        self.units=units

    def shot_points(self) -> pd.DataFrame:
        """Extract shot-point navigation from the SEG-Y trace headers.
        
        Returns
        -------
        pd.DataFrame
            Columns: ``File_name``, ``Trace``, ``Shotpoint``, ``Lat``, ``Long``.
        """
        r = self.parent.reader
        sc = np.array(r.attributes(self.parent.byte_position.SourceGroupScalar)).astype("float")
        self.scalar = sc.copy()
        sc[sc==0]=1
        sc[sc < 0] = 1.0 / abs(sc[sc < 0])

        units = np.array(r.attributes(self.parent.byte_position.CoordinateUnits))
        units[units==0]=2
        long = np.array(r.attributes(self.parent.byte_position.SourceX)) * sc
        lat = np.array(r.attributes(self.parent.byte_position.SourceY)) * sc
        shotpoints = np.array(r.attributes(self.parent.byte_position.EnergySourcePoint))
        trace_no = np.array(r.attributes(self.parent.byte_position.TRACE_SEQUENCE_FILE))

        nav_df = pd.DataFrame(
            {
                "File_name": [self.parent.meta["basename"]] * self.parent.meta["n_traces"],
                "Trace": trace_no,
                "Shotpoint": shotpoints,
                "Lat": lat,
                "Long": long,
            }
        )
        self.shotpoints = nav_df
        self.units = units
        return nav_df
        
    def coords_2_utm(
        self,
        src_projection: Optional[str] = None,
        out_projection: Optional[str] = None,
    ) -> None:
        """Reproject shot-point coordinates to a target CRS.

        Parameters
        ----------
        src_projection : str, optional
            Source EPSG code. Required when the source coordinates are in UTM.
        out_projection : str
            Target EPSG code (e.g. ``'EPSG:32618'`` for UTM zone 18N).
        """
        if not self.coords_converted:
            if self.shotpoints is None:
                self.shot_points()
    
            df = self.shotpoints
            lon, lat, m = decode_nav(df["Long"].values, df["Lat"].values, self.units, 1)
    
            if not m.any():   # all decimal degrees
                transformer = pyproj.Transformer.from_crs("EPSG:4326", out_projection, always_xy=True)
            else:
                transformer = pyproj.Transformer.from_crs(src_projection, out_projection, always_xy=True)
    
            utm_x_all, utm_y_all = transformer.transform(lon, lat)
            self.shotpoints["Long"] = (utm_x_all * 100).astype("int")
            self.shotpoints["Lat"] = (utm_y_all * 100).astype("int")
            self.units = np.ones_like(lon)
            self.scalar = np.ones_like(lon) * -100
            self.coords_converted = True
            self.coords_prj = out_projection
            
    def to_shp(
        self,
        outfile: str,
        line: bool = True,
        append_if_exist: bool = True,
        epsg_code: Optional[Union[int, str]] = None,
    ) -> None:
        """Export shot-point navigation to a shapefile.

        Parameters
        ----------
        outfile : str
            Output path (the ``.shp`` extension is appended if missing).
        line : bool
            Export a line shapefile (default).  Pass ``False`` for points.
        append_if_exist : bool
            Append to an existing file of the same name.  When ``False``
            the new file is renamed to avoid overwriting.
        epsg_code : int or str, optional
            Required when any traces are in UTM; used to convert to
            WGS-84 decimal degrees before writing.
        """
        if not outfile.endswith(".shp"):
            outfile += ".shp"

        if self.shotpoints is None:
            self.shot_points()
        # if crop function was called
        strt = self.parent.meta['good_traces'][0]
        nd = self.parent.meta['good_traces'][-1]
        
        df = self.shotpoints
        #df = df.iloc[strt:nd, :]
        
        lon, lat, m = decode_nav(df["Long"].values[strt:nd], df["Lat"].values[strt:nd], self.units[strt:nd], 1)

        if m.any():
            if epsg_code is None:
                raise ValueError(
                    "Some/all traces are in UTM coordinates; "
                    "provide an epsg_code for conversion to decimal degrees."
                )
            lat[m], lon[m] = utm_to_decimal_degrees(lon[m], lat[m], epsg_code)

        append = False
        if os.path.exists(outfile):
            if append_if_exist:
                append = True
            else:
                outfile = outfile[:-4] + "_1.shp"

        if line:
            coords = list(zip(lon, lat))
            geom = LineString(coords) if len(coords) >= 2 else None
            attr = {
                "ID": df.loc[0, "File_name"],
                "SP_start": df.iloc[strt, 2],
                "SP_end": df.iloc[nd, 2],
            }
            new_gdf = gpd.GeoDataFrame([attr], geometry=[geom], crs="EPSG:4326")
        else:
            geometry = gpd.points_from_xy(lon, lat)
            attr = {"ID": df["File_name"].values[strt:nd], "Shotpoint": df.iloc[:, 2].values[strt:nd]}
            new_gdf = gpd.GeoDataFrame(attr, geometry=geometry, crs="EPSG:4326")

        if append:
            print(f"'{outfile}' exists — appending data…")
            existing_gdf = gpd.read_file(outfile)
            if existing_gdf.crs != new_gdf.crs:
                new_gdf = new_gdf.to_crs(existing_gdf.crs)
            new_gdf = gpd.GeoDataFrame(
                pd.concat([existing_gdf, new_gdf], ignore_index=True),
                crs=existing_gdf.crs,
            )

        new_gdf.to_file(outfile, driver="ESRI Shapefile")
        print(f"'{outfile}' exported successfully.")
