import warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


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