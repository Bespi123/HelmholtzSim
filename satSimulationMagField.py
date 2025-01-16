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
    Initialize the satellite and transformer as global variables.
    """
    global satellite, transformer
    satellite = Satrec.twoline2rv(line1, line2)
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

def calculate_gmst(jd_full):
    """
    Calculate GMST (Greenwich Mean Sidereal Time).
    """
    T = (jd_full - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd_full - 2451545.0) + T**2 * (0.000387933 - T / 38710000.0)
    return np.radians(gmst_deg % 360)

def process_time(current_time):
    """
    Process a single timestamp for satellite propagation and field calculations.
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

def simulate_satellite(start_date, end_date, time_step):
    """
    Simulate satellite propagation over a given time range with parallel processing.
    """
    # Generate a range of timestamps
    time_range = [start_date + i * time_step for i in range(int((end_date - start_date) / time_step) + 1)]

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(filter(None, executor.map(process_time, time_range)))

    # Convert results to DataFrame
    return pd.DataFrame(results)