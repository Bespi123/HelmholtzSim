from concurrent.futures import ProcessPoolExecutor
from pyproj import Transformer
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from wmm2020 import wmm

# Global variables
satellite = None
transformer = None

def initialize_satellite(line1, line2):
    """
    Initialize the satellite and coordinate transformer as global variables.

    Parameters:
    - line1 (str): First line of the Two-Line Element (TLE) set.
    - line2 (str): Second line of the Two-Line Element (TLE) set.

    This function initializes:
    1. `satellite`: An SGP4 propagator object created using the TLE.
    2. `transformer`: A coordinate transformation object to convert ECEF (Earth-Centered, Earth-Fixed)
       coordinates to geodetic (latitude, longitude, altitude) using EPSG codes.

    The initialized objects are stored as global variables for use in other functions.
    """
    global satellite, transformer
    satellite = Satrec.twoline2rv(line1, line2)
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

def calculate_gmst(jd_full):
    """
    Calculate Greenwich Mean Sidereal Time (GMST) in radians.

    Parameters:
    - jd_full (float): The full Julian Date (JD) for which GMST is to be calculated.

    Returns:
    - float: GMST in radians.
    """
    T = (jd_full - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd_full - 2451545.0) + T**2 * (0.000387933 - T / 38710000.0)
    return np.radians(gmst_deg % 360)

def process_time(current_time):
    """
    Process a single timestamp for satellite propagation and magnetic field calculations.

    Parameters:
    - current_time (datetime): The timestamp for which to compute satellite position and field values.

    Returns:
    - dict: A dictionary containing satellite position (ECI, ECEF, geodetic) and magnetic field components.
    - None: If an error occurs during propagation.
    """
    jd, fr = jday(current_time.year, current_time.month, current_time.day,
                  current_time.hour, current_time.minute, current_time.second)

    error_code, position, velocity = satellite.sgp4(jd, fr)

    if error_code != 0:
        return None

    x_eci, y_eci, z_eci = position
    gmst = calculate_gmst(jd + fr)

    # Convert ECI to ECEF
    x_ecef = x_eci * np.cos(gmst) + y_eci * np.sin(gmst)
    y_ecef = -x_eci * np.sin(gmst) + y_eci * np.cos(gmst)
    z_ecef = z_eci

    # Convert ECEF to geodetic
    lon, lat, alt = transformer.transform(x_ecef * 1e3, y_ecef * 1e3, z_ecef * 1e3)

    # Calculate magnetic field
    field = wmm(lon, lat, alt / 1000, current_time.year)
    north, east, down = field.north[0, 0].values.item(), field.east[0, 0].values.item(), field.down[0, 0].values.item()

    # Convert magnetic field components to ECEF
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bx_ecef = (-np.sin(lat_rad) * np.cos(lon_rad) * north -
               np.sin(lon_rad) * east -
               np.cos(lat_rad) * np.cos(lon_rad) * down)
    by_ecef = (-np.sin(lat_rad) * np.sin(lon_rad) * north +
               np.cos(lon_rad) * east -
               np.cos(lat_rad) * np.sin(lon_rad) * down)
    bz_ecef = np.cos(lat_rad) * north - np.sin(lat_rad) * down

    # Rotate from ECEF to ECI
    bx_eci = bx_ecef * np.cos(gmst) - by_ecef * np.sin(gmst)
    by_eci = bx_ecef * np.sin(gmst) + by_ecef * np.cos(gmst)
    bz_eci = bz_ecef

    # Return results as a dictionary
    return {
        "Time (UTC)": current_time,
        "ECI X (km)": x_eci,
        "ECI Y (km)": y_eci,
        "ECI Z (km)": z_eci,
        "ECEF X (km)": x_ecef,
        "ECEF Y (km)": y_ecef,
        "ECEF Z (km)": z_ecef,
        "Latitude (deg)": lat,
        "Longitude (deg)": lon,
        "Altitude (m)": alt,
        "B N": north,
        "B E": east,
        "B D": down,
        "Bx ECEF (nT)": bx_ecef,
        "By ECEF (nT)": by_ecef,
        "Bz ECEF (nT)": bz_ecef,
        "Bx ECI (nT)": bx_eci,
        "By ECI (nT)": by_eci,
        "Bz ECI (nT)": bz_eci
    }

def simulate_satellite(start_date, end_date, time_step, batch_size=100):
    """
    Simulate satellite propagation over a given time range in batches with parallel processing.

    Parameters:
    - start_date: datetime, start of the simulation
    - end_date: datetime, end of the simulation
    - time_step: timedelta, time interval between steps
    - batch_size: int, number of timestamps processed in each batch

    Returns:
    - DataFrame containing results
    """
    # Ensure time_step is in seconds
    time_step_seconds = time_step.total_seconds()

    # Generate a range of timestamps
    num_steps = int((end_date - start_date).total_seconds() / time_step_seconds) + 1
    time_range = [start_date + timedelta(seconds=i * time_step_seconds) for i in range(num_steps)]

    # Split the time_range into batches
    time_batches = np.array_split(time_range, max(1, len(time_range) // batch_size))

    results = []
    with ProcessPoolExecutor() as executor:
        for batch in time_batches:
            batch_results = list(filter(None, executor.map(process_time, batch)))
            results.extend(batch_results)  # Append batch results

    # Convert results to DataFrame
    return pd.DataFrame(results)


def calculate_max_min_values(df, select):
    """
    Calculate and print the maximum and minimum values for specified columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - columns (list of str): List of column names to process.

    Returns:
    - results (dict): Dictionary containing max and min values for each column.
    """
    results = {}

    if select == 'ECI':
        columns = ['Bx ECI (nT)', 'By ECI (nT)', 'Bz ECI (nT)']
    elif select == 'ECEF':
        columns = ['Bx ECEF (nT)', 'By ECEF (nT)', 'Bx ECEF (nT)']
    elif select == 'NED':
        columns = ['B N', 'B E', 'B D']
    else :
       print('Not supported reference frame')
    
    for col in columns:
        max_value = df[col].max()
        min_value = df[col].min()
        
        results[col] = {"max": max_value, "min": min_value}
        print(f"Max Value {col}: {max_value} nT, Min Value {col}: {min_value} nT")
    
    return results