from utils import *

Num_data_points = 50000  # Use all data
numpy_dtype = np.float64
obs, dt, gdf = get_taxi_data(Num_data_points, dtype=numpy_dtype)
