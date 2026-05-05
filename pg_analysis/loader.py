import warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pg_analysis import plotter as pga
import json



class Loader:
    """
    Non-interactive data loader that:
    - Reads tabular files into a pandas DataFrame
    - Attaches units per column
    - Retains filename metadata
    - Optionally enforces strict unit completeness
    """

    # Default reference units - these cover most worm tracker cases in ScholzLab
    UNITS: Dict[str, str] = {
        'x': 'px',
        'y': 'px',
        'x_scaled': 'um',
        'y_scaled': 'um',
        'frame': '1',
        'time': 's',
        'time_align': 's',
        'time_aligned': 's',
        'pumps': 'a.f.u.',
        'pumps_clean': 'a.f.u.',
        'pump_events': '1',
        'rate': '1/s',
        'count_rate': '1/s',
        'count_rate_pump_events': '1/s',
        'velocity': 'um/s',
        'velocity_smooth': 'um/s',
        'nose_speed': 'um/s',
        'cms_speed': 'um/s',
        'reversals': '1',
        'reversals_nose': '1',
        'inside': '1',
        'Imean': 'a.f.u.',
        'Imax': 'a.f.u.',
        'Istd': 'a.f.u.',
        'skew': '1',
        'area': 'px^2',
        'Area2': 'px^2',
        'size': 'mm',
        'Centerline': '1',
        'centerline_scaled': 'um',
        'Straightened': '1',
        'temperature': 'C',
        'humidity': '%',
        'age': 'h',
        '@acclimation': 'min',
        'particle': '1',
        'image_index': '1',
        'im_idx': '1',
        'has_image': '1',
        'index': '1',
        'space_units': 'um',
        'time_units': 's',
    }


    SUPPORTED_FORMATS = {".csv", ".xls", ".xlsx", ".parquet", ".json"}

    def __init__(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
        units: Optional[Dict[str, str]] = None,
        strict_units: bool = False,
        strict_columns: bool = True,
        **read_kwargs
    ):
        self.filepath = Path(filepath)
        self.filename = self.filepath.name

        self.centerline = None
        self.images = None

        self.df = self._load_file(columns, strict_columns, **read_kwargs)

        self.units = self._resolve_units(units or {}, strict_units)

        self.df.attrs["filename"] = self.filename
        self.df.attrs["units"] = self.units

    def _load_file(
        self,
        columns: Optional[List[str]],
        strict_columns: bool,
        **kwargs
    ) -> pd.DataFrame:

        suffix = self.filepath.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}")

        # --- CSV: use usecols directly (memory efficient) ---
        if suffix == ".csv":
            if columns is not None:
                df = pd.read_csv(
                    self.filepath,
                    usecols=columns,
                    **kwargs
                )
                # Pandas already raises if columns missing,
                # but we optionally enforce custom error message
                if strict_columns:
                    missing = [c for c in columns if c not in df.columns]
                    if missing:
                        raise ValueError(
                            f"Requested columns not found: {missing}"
                        )
            else:
                df = pd.read_csv(self.filepath, **kwargs)

        # --- Other formats ---
        elif suffix in {".xls", ".xlsx"}:
            df = pd.read_excel(self.filepath, **kwargs)

        elif suffix == ".parquet":
            df = pd.read_parquet(self.filepath, columns=columns, **kwargs)

        elif suffix == ".json":
            df = pd.read_json(self.filepath, **kwargs)

        # Post-filter for non-CSV formats (if needed)
        if suffix != ".csv" and columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing and strict_columns:
                raise ValueError(f"Requested columns not found: {missing}")

            df = df[[c for c in columns if c in df.columns]]
        # extract the non-scalar data if present
        if "Centerline" in df.columns:
            if isinstance(df['Centerline'][0], str):
                self.centerline = np.array(
                [np.array(json.loads(cl)) for cl in df["Centerline"]]
                )
            else:
                self.centerline = np.array(
                    [np.array(cl) for cl in df["Centerline"]]
                )

        if "Straightened" in df.columns:
            self.images = np.array(
                [np.array(im) for im in df["Straightened"]]
            )
        # Drop structured columns from dataframe
        df = df.drop(["Centerline", "Straightened"], errors="ignore")
        self.df = df
        return df

    def _resolve_units(
        self,
        user_units: Dict[str, str],
        strict: bool
    ) -> Dict[str, str]:
        """Resolve units assignment to columns."""
        resolved = {}
        missing_columns = []

        for col in self.df.columns:
            if col in user_units:
                resolved[col] = user_units[col]
            elif col in self.UNITS:
                resolved[col] = self.UNITS[col]
            else:
                if strict:
                    missing_columns.append(col)
                else:
                    resolved[col] = "dimensionless"

        if strict and missing_columns:
            raise ValueError(
                f"No unit specified for columns: {missing_columns}"
            )

        return resolved

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def get_units(self) -> Dict[str, str]:
        return self.units
    
    def get_centerline(self) -> np.array:
        return self.centerline
    
    def get_images(self) -> np.array:
        return self.images

    def __repr__(self) -> str:
        return (
            f"Loader(filename='{self.filename}', "
            f"columns={list(self.df.columns)}, "
            f"strict_units={len(self.units) == len(self.df.columns)})"
        )


class MacroscopeRawStageLoader(Loader):
    """
    Loader subclass for Macroscope stage files. This assumes the structure from GlowTracker output.
    - skips header rows
    - fixes misaligned columns
    - renames columns
    - resets time to zero
    """
    UNITS: Dict[str, str] = {
        "frame":1,
        "Time": 'us',
        "x": 'mm',
        "y":'mm',
        "z":'mm',
        "minBrightness":'a.f.u',
        "maxBrightness":'a.f.u',
        "meanBrightness":'a.f.u',
        "medianBrightness":'a.f.u',
        "skewness":'a.f.u',
        "percentile_5":'a.f.u',
        "percentile_95":'a.f.u',
        "time_units":'s',
        "space_units":'um'
    }
    def _load_file(self, 
                   columns: Optional[List[str]],
                strict_columns: bool,**kwargs) -> pd.DataFrame:
        """
        Overrides Loader._load_file to handle Macroscope stage CSVs.
        """
        fname = str(self.filepath)

        # --- Load raw stage data ---
        df = pd.read_csv(
            fname,
            skiprows=27,
            sep=' ',
            index_col=False,
            comment=None,
            **kwargs
        )

        # Shift columns: comment recognized as first column
        cols = df.columns[1:]
        df = df.iloc[:, :-1]
        df.columns = cols

        # Keep only requested columns
        #usecols = ['Frame', 'Time', 'X', 'Y']

        # Rename columns to standard format
        df = df.rename(columns={
            'Frame': 'frame',
            'Time': 'time',
            'X': 'x',
            'Y': 'y',
        })

        # Reset time to start at zero -- macroscope uses system time
        df['time'] = df['time'] - df['time'].iloc[0]

        # Optional: apply column selection from Loader
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing and strict_columns:
                raise ValueError(f"Requested columns not found: {missing}")
            df = df[[c for c in columns if c in df.columns]]


        return df
    
    
    
class MacroscopeLoader(Loader):
    """
    Loader subclass for files created by MacroscopeDataAnalysis, the signals and centerlines.
    """
    UNITS: Dict[str, str] = {
        "frame":1,
        "Time": 'us',
        "x": 'um',
        "y":'um',
        "z":'um',
        "minBrightness":'a.f.u',
        "maxBrightness":'a.f.u',
        "meanBrightness":'a.f.u',
        "medianBrightness":'a.f.u',
        "skewness":'a.f.u',
        "percentile_5":'a.f.u',
        "percentile_95":'a.f.u',
        "Xstage": 'um',
        "Ystage": 'um',
        "Xworm":'um',
        "Yworm":'um',
        "signal_max": 'a.f.u',
        "signal_mean": 'a.f.u',
        "cms_y":'um',
        "cms_x":'um',
        "skew":'a.f.u',
        "time":'s',
        'space_units': 'um',
        'time_units': 's',
        
    }
    def _load_file(self, 
                   columns: Optional[List[str]],
                   
                strict_columns: bool,**kwargs) -> pd.DataFrame:
        """
        Overrides Loader._load_file to handle Macroscope stage CSVs.
        """
        fname = str(self.filepath)

        # --- Load raw stage data ---
        df = pd.read_json(fname, orient='split')
         # Optional: apply column selection from Loader
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing and strict_columns:
                raise ValueError(f"Requested columns not found: {missing}")
            df = df[[c for c in columns if c in df.columns]]

        # ---- find associated centerline file. This assumes namimg convention as in macroscope_data analysis
        try:
            stem = Path(fname).stem.removesuffix('_signals')
            fname_cl = Path(fname).parent / (stem + '_um_centerlines.csv')
            self.centerline = np.loadtxt(fname_cl, delimiter = ',').reshape(-1,100,2)
        except FileNotFoundError:
            warnings.warn(f'No centerline file found at {fname_cl}')
       
        return df
    

    
class MultiWormLoaderTrackMate:
    """
    Reads a multi-particle CSV once and vends one Loader-compatible
    object per particle.
    """
    # rename columns
    
    def __init__(
        self,
        filepath: str,
        particle_col: str = "particle",
        columns: Optional[List[str]] = None,
        sort_column = 'FRAME'
    ):

        rename_map = {
            "LABEL": "label",
            "ID": "id",
            "QUALITY": "quality",
            "POSITION_X": "x",
            "POSITION_Y": "y",
            "POSITION_Z": "z",
            "POSITION_T": "t",
            "FRAME": "frame",
            "RADIUS": "radius",
            "VISIBILITY": "visibility",
            "MEAN_INTENSITY_CH1": "mean_intensity_ch1",
            "MEDIAN_INTENSITY_CH1": "median_intensity_ch1",
            "MIN_INTENSITY_CH1": "min_intensity_ch1",
            "MAX_INTENSITY_CH1": "max_intensity_ch1",
            "TOTAL_INTENSITY_CH1": "total_intensity_ch1",
            "STD_INTENSITY_CH1": "std_intensity_ch1",
            "CONTRAST_CH1": "contrast_ch1",
            "SNR_CH1": "snr_ch1",
        }
        self.filepath = Path(filepath)
        self.particle_col = particle_col
        self.sort_column = rename_map.get(sort_column, sort_column)
        # read units
        header = pd.read_csv(self.filepath, nrows=0).columns.tolist()
        units = pd.read_csv(self.filepath, skiprows=3, nrows=1, header=None).iloc[0].tolist()
        units = [u.strip("()") if isinstance(u, str) else u for u in units]
        
        self.units = dict(zip(header, units))
        self.units = {rename_map.get(k,k): v for k, v in self.units.items()}
        
        df = pd.read_csv(filepath, skiprows=(1,2,3))
        if particle_col not in df.columns:
            raise ValueError(f"Column '{particle_col}' not found in {filepath}")
        if columns:
            keep = list({particle_col} | (set(columns) & set(df.columns)))
            df = df[keep]
        
        
        # Build mapping for actual columns in the dataframe
        actual_rename = {}
        for old_col in df.columns:
            if old_col in rename_map:
                actual_rename[old_col] = rename_map[old_col]
        df = df.rename(columns=actual_rename)
        # group by track
        self._groups = {
            pid: grp.reset_index(drop=True)
            for pid, grp in df.groupby(particle_col)
        }

    @property
    def particle_ids(self) -> List:
        return list(self._groups.keys())

    def get_particle_loader(self, particle_id) -> "_SingleParticleLoader":
        if particle_id not in self._groups:
            raise KeyError(f"Particle {particle_id} not found.")
        return _SingleParticleLoader(self._groups[particle_id], self.units, self.sort_column)



def worms_from_multi_csv(
    filepath: str,
    fps: float,
    scale: float,
    particle_col: str = "particle",
    columns: Optional[List[str]] = None,
    scale_units: str = "um",
    fps_units: str = "s",
    particle_ids: Optional[List] = None,   # ← add this
) -> List[Worm]:
    """
    Factory: read a multi-particle CSV and return one Worm per particle - uses the normal loader class.

    Example
    -------
    worms = worms_from_multi_csv("tracks.csv", fps=25, scale=0.16)
    for w in worms:
        w.load_data()
    """
    multi = MultiWormLoaderTrackMate(filepath, particle_col=particle_col, columns=columns)
    # allows to filter ids
    ids = particle_ids if particle_ids is not None else multi.particle_ids
    return [
        pga.Worm(
            filename=filepath,
            fps=fps,
            scale=scale,
            scale_units=scale_units,
            fps_units=fps_units,
            columns=columns,
            particle_index=pid,
            loader_cls=None,
            loader_kwargs={},
            preloaded_loader=multi.get_particle_loader(pid),  # ← picked up by load_data
        )
        for pid in ids
        if pid in multi.particle_ids 
    ]


class _SingleParticleLoader:
    """Loader-compatible shim for a single particle's DataFrame slice."""

    def __init__(self, df: pd.DataFrame, units: dict, sort_column = 'frame'):
        self._df = df.sort_values(sort_column).reset_index()
        self.units = units
        self.centerline = None
        self.images = None

    def get_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def get_units(self) -> dict:
        
        return self.units