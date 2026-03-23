import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from math import radians, sin, cos, sqrt, atan2

cities = {
    'brussels': (50.8503, 4.3517),
    'antwerp':  (51.2194, 4.4025),
    'ghent':    (51.0543, 3.7174),
    'liege':    (50.6326, 5.5797),
    'bruges':   (51.2093, 3.2247),
    'namur':    (50.4674, 4.8719),
    'leuven':   (50.8798, 4.7005),
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_distances(locality: str, distances_path: str = 'data/distances.csv') -> dict:
    """
    Retourne les distances d'une localité vers les grandes villes.
    Cherche dans le CSV, géocode et sauvegarde si manquante.
    """
    distances = pd.read_csv(distances_path)

    # Localité déjà connue → retourner directement
    row = distances[distances['Locality'].str.lower() == locality.lower()]
    if not row.empty:
        return row.iloc[0].to_dict()

    # Localité manquante → géocoder
    print(f"'{locality}' non trouvée dans le CSV, géocodage...")
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="my_app")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location   = geocode(f"{locality}, Belgium")

    if not location:
        raise ValueError(f"Localité introuvable : {locality}")

    new_row: dict[str, str | float] = {'Locality': locality}
    for city, (lat, lon) in cities.items():
        new_row[f'dist_{city}'] = haversine(location.latitude, location.longitude, lat, lon)

    # Ajouter au CSV
    distances = pd.concat([distances, pd.DataFrame([new_row])], ignore_index=True)
    distances.to_csv(distances_path, index=False)
    print(f"'{locality}' ajoutée au CSV.")

    return new_row