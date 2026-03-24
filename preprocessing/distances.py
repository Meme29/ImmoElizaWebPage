import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from fastapi import HTTPException

cities = {
    'brussels': (50.8503, 4.3517),
    'antwerp':  (51.2194, 4.4025),
    'ghent':    (51.0543, 3.7174),
    'liege':    (50.6326, 5.5797),
    'bruges':   (51.2093, 3.2247),
    'namur':    (50.4674, 4.8719),
    'leuven':   (50.8798, 4.7005),
}

coast_points = [
    (51.1120, 2.5230),  # De Panne
    (51.1165, 2.5986),  # Koksijde
    (51.1333, 2.6500),  # Oostduinkerke
    (51.1500, 2.7200),  # Middelkerke
    (51.2254, 2.9175),  # Oostende
    (51.2833, 3.0167),  # De Haan
    (51.3000, 3.1333),  # Blankenberge
    (51.3167, 3.1833),  # Zeebrugge
    (51.3620, 3.3650),  # Knokke-Heist
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_distances(locality: str, distances_path: str = 'data/distances.csv') -> dict:
    distances = pd.read_csv(distances_path)

    row = distances[distances['Locality'].str.lower() == locality.lower()]
    if not row.empty:
        return row.iloc[0].to_dict()

    print(f"'{locality}' non trouvée dans le CSV, géocodage...")
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="my_app")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location   = geocode(f"{locality}, Belgium")

    if not location:
        raise ValueError(f"Localité introuvable : {locality}")

    lat, lon = location.latitude, location.longitude

    new_row: dict[str, str | float] = {'Locality': locality}
    for city, (clat, clon) in cities.items():
        new_row[f'dist_{city}'] = haversine(lat, lon, clat, clon)
    new_row['dist_sea'] = min(haversine(lat, lon, clat, clon) for clat, clon in coast_points)

    distances = pd.concat([distances, pd.DataFrame([new_row])], ignore_index=True)
    distances.to_csv(distances_path, index=False)
    print(f"'{locality}' ajoutée au CSV.")

    return new_row

_postal_df = None

def _get_postal_df():
    global _postal_df
    if _postal_df is None:
        df = pd.read_csv('data/postal_codes_clean.csv')
        df['locality_norm'] = df['locality'].str.lower().str.strip()
        _postal_df = df
    return _postal_df

def validate_locality_zip(locality: str, zip_code: int) -> str | None:
    df = _get_postal_df()

    matches_zip = df[df['zip_code'] == zip_code]
    if matches_zip.empty:
        return f"ZIP code {zip_code} not found."

    locality_norm = locality.lower().strip()
    match = matches_zip[matches_zip['locality_norm'] == locality_norm]

    if match.empty:
        valid = matches_zip['locality'].tolist()
        raise HTTPException(status_code=400, detail=f"Locality '{locality}' does not match ZIP code {zip_code}. Valid localities: {', '.join(valid)}.")
    return None