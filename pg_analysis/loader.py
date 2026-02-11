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

    def __repr__(self) -> str:
        return (
            f"Loader(filename='{self.filename}', "
            f"columns={list(self.df.columns)}, "
            f"strict_units={len(self.units) == len(self.df.columns)})"
        )