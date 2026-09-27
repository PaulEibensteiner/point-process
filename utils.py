from functools import wraps
from pathlib import Path
import time
import zipfile
import geopandas
import numpy as np

import pandas as pd
import os
import torch
import importlib.resources

data_path = Path(__file__).parent / "data"


if not os.path.exists(data_path):
    os.makedirs(data_path)


def get_grid(discretization, left, down, right, up):
    # Create evenly spaced values
    x = torch.linspace(left, right, discretization)
    y = torch.linspace(down, up, discretization)

    # Create a grid of 2D points
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    # Stack the grid to create a tensor of shape (discretization, discretization, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # Reshape the grid to have shape (discretization * discretization, 2)
    xtest = grid.view(-1, 2)
    return xtest


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.process_time()
        result = f(*args, **kw)
        te = time.process_time()
        print(f"func:{f.__name__} args:{args} took: {te-ts:.4f} sec")
        return result

    return wrap


""" 
def kaggle_download(path: Path, name: str):
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(path):
        os.makedirs(path)

    # Create the download directory if it doesn't exist

    if not os.path.exists(path / "train.csv"):
        api.competition_download_files(
            name,
            path=path,
        )

        with zipfile.ZipFile(path / f"{name}.zip", "r") as zip_ref:
            zip_ref.extractall(path)

        os.remove(path / "pkdd-15-predict-taxi-service-trajectory-i.zip")
        # unzip every .zip in path
        for file in os.listdir(path):
            if file.endswith(".zip"):
                with zipfile.ZipFile(path / file, "r") as zip_ref:
                    zip_ref.extractall(path)
                os.remove(path / file)
                
"""


def get_taxi_data(
    subsample: int | None, dtype=np.float64
) -> tuple[list, geopandas.GeoDataFrame]:
    with importlib.resources.open_text(
        "sensepy.benchmarks.data", "taxi_data.csv"
    ) as file:
        df = pd.read_csv(file)

    df = df[df["Longitude"] < -8.580]
    df = df[df["Longitude"] > -8.64]
    df = df[df["Latitude"] > 41.136]
    df = df[df["Latitude"] < 41.17]
    if subsample is not None:
        df = df.head(subsample)

    g = geopandas.points_from_xy(df.Longitude, df.Latitude)
    gdf = geopandas.GeoDataFrame(df, geometry=g)  # type: ignore
    gdf.crs = "EPSG:4326"

    # cleaning nans
    obs = df.values[:, [1, 2]].astype(dtype)
    obs = obs[~np.isnan(obs)[:, 0], :]

    x_max = np.max(obs[:, 0])  # longitude
    x_min = np.min(obs[:, 0])

    y_max = np.max(obs[:, 1])  # lattitude
    y_min = np.min(obs[:, 1])
    lat = df["Latitude"]
    long = df["Longitude"]

    left, right = long.min(), long.max()
    down, up = lat.min(), lat.max()

    # transform from map to [-1,1]
    transform_x = lambda x: (2 / (x_max - x_min)) * x + (
        1 - (2 * x_max / (x_max - x_min))
    )
    transform_y = lambda y: (2 / (y_max - y_min)) * y + (
        1 - (2 * y_max / (y_max - y_min))
    )

    # transform from [-1,1] to map
    inv_transform_x = lambda x: (x_max - x_min) / 2 * x + (x_min + x_max) / 2
    inv_transform_y = lambda x: (y_max - y_min) / 2 * x + (y_min + y_max) / 2

    # transform to [-1,1]
    obs[:, 0] = np.apply_along_axis(transform_x, 0, obs[:, 0])
    obs[:, 1] = np.apply_along_axis(transform_y, 0, obs[:, 1])

    # extract temporal information of the dataset
    df["Date"] = pd.to_datetime(df["Date"])
    # time section of the dataset in minutes
    dt = (df["Date"].max() - df["Date"].min()).seconds // 60
    # This data apparently is one single sample??
    return torch.from_numpy(obs), dt, gdf


def _fix_mojibake(value):
    """Most strings of the WFS are UTF-8 that was decoded as Latin-1 or cp1252."""
    if not isinstance(value, str):
        return value
    try:
        raw = b"".join(
            c.encode("cp1252") if ord(c) > 255 else bytes([ord(c)]) for c in value
        )
        return raw.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


ZUERI_WIE_NEU_WFS = (
    "https://www.ogd.stadt-zuerich.ch/wfs/geoportal/Zueri_wie_neu"
    "?service=WFS&version=1.1.0&request=GetFeature&typename=zwn_meldungen_p"
    "&outputFormat=GeoJSON&srsName=EPSG:4326"
)


def get_zueri_wie_neu_data(
    subsample: int | None, dtype=np.float64, service_code: str | None = None
) -> tuple[torch.Tensor, float, geopandas.GeoDataFrame]:
    """Reports of damaged infrastructure in Zurich
    (https://data.stadt-zuerich.ch/dataset/geo_zueri_wie_neu).

    Returns observations scaled to [-1,1]^2, the observed time span in days
    and the GeoDataFrame in EPSG:4326.
    """
    path = data_path / "zueri_wie_neu.geojson"
    if not os.path.exists(path):
        import urllib.request

        urllib.request.urlretrieve(ZUERI_WIE_NEU_WFS, path)

    gdf = geopandas.read_file(path)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    for column in gdf.select_dtypes(include=["object", "string"]).columns:
        gdf[column] = gdf[column].map(_fix_mojibake)
    if service_code is not None:
        gdf = gdf[gdf["service_code"] == service_code]
    if subsample is not None:
        gdf = gdf.head(subsample)
    gdf = gdf.reset_index(drop=True)

    obs = np.stack([gdf.geometry.x.values, gdf.geometry.y.values], axis=1).astype(
        dtype
    )
    mins, maxs = obs.min(axis=0), obs.max(axis=0)
    # transform from map to [-1,1]
    obs = 2 * (obs - mins) / (maxs - mins) - 1

    # time section of the dataset in days
    t = pd.to_datetime(gdf["requested_datetime"])
    dt = (t.max() - t.min()).total_seconds() / (60 * 60 * 24)
    return torch.from_numpy(obs), dt, gdf


WEB_MERCATOR_RADIUS = 6378137.0


def lonlat_to_mercator(lon, lat):
    """EPSG:4326 degrees to EPSG:3857 meters."""
    x = WEB_MERCATOR_RADIUS * np.deg2rad(lon)
    y = WEB_MERCATOR_RADIUS * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))
    return x, y


def mercator_to_lonlat(x, y):
    """EPSG:3857 meters to EPSG:4326 degrees."""
    lon = np.rad2deg(x / WEB_MERCATOR_RADIUS)
    lat = np.rad2deg(2 * np.arctan(np.exp(y / WEB_MERCATOR_RADIUS)) - np.pi / 2)
    return lon, lat
