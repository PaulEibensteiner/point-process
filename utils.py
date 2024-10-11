from functools import wraps
from pathlib import Path
import time
import zipfile
import geopandas
import numpy as np

import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from stpy.borel_set import BorelSet
import torch
import importlib.resources

data_path = Path(__file__).parent / "data"


if not os.path.exists(data_path):
    os.makedirs(data_path)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.process_time()
        result = f(*args, **kw)
        te = time.process_time()
        print(f"func:{f.__name__} args:{args} took: {te-ts:.4f} sec")
        return result

    return wrap


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


def get_taxi_data(subsample: int, D: BorelSet) -> tuple[list, geopandas.GeoDataFrame]:
    with importlib.resources.open_text(
        "sensepy.benchmarks.data", "taxi_data.csv"
    ) as file:
        df = pd.read_csv(file)

    df = df[df["Longitude"] < -8.580]
    df = df[df["Longitude"] > -8.64]
    df = df[df["Latitude"] > 41.136]
    df = df[df["Latitude"] < 41.17]
    df = df.head(subsample)

    g = geopandas.points_from_xy(df.Longitude, df.Latitude)
    gdf = geopandas.GeoDataFrame(df, geometry=g)  # type: ignore
    gdf.crs = "EPSG:4326"

    # cleaning nans
    obs = df.values[:, [1, 2]].astype(float)
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
    return [(D, torch.from_numpy(obs), dt)], gdf
