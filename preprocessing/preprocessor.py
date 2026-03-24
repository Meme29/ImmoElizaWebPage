from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from .distances import get_distances, cities, validate_locality_zip, _get_postal_df

router = APIRouter()

class PropertyInput(BaseModel):
    area:             int   # Non null
    property_type:    str   # Non null
    subtype:          str   # Non null
    rooms_number:     int   # Non null
    zip_code:         int   # Non null
    locality:         str   # Non null
    land_area:        int   = 0
    garden:           bool  = False
    garden_area:      int   = 0
    equipped_kitchen: bool  = False  # OUT
    swimming_pool:    bool  = False
    furnished:        bool  = False
    open_fire:        bool  = False  # OUT
    terrace:          bool  = False
    terrace_area:     int   = 0
    facades_number:   int   = 0
    building_state:   str   = 'GOOD'

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])
    
@router.get("/localities")
def get_all_localities():
    df = _get_postal_df()
    result = df[['locality', 'zip_code']].drop_duplicates(subset=['locality']).dropna().sort_values('locality')
    return {"localities": result.to_dict(orient='records')}


@router.post("", tags=["Preprocessor"])
def clean(property: PropertyInput):
    df = property.to_df()

    # ----------------------------- #
    #       Drop useless columns    #
    # ----------------------------- #
    df.drop(columns=['terrace_area', 'garden_area', 'open_fire', 'equipped_kitchen'], inplace=True)

    # ----------------------------- #
    #   Validation localité / ZIP   #
    # ----------------------------- #
    error = validate_locality_zip(property.locality, property.zip_code)
    if error:
        print("Validation error:", error)
        raise HTTPException(status_code=400, detail="Locality doesn't match ZIP code.")

    # ----------------------------- #
    #   Type of property mapping    #
    # ----------------------------- #
    subtype_to_type = {
        'Mixed Building': 'Apartment',
        'Cottage':        'House',
        'Master House':   'House',
        'Triplex':        'Apartment',
    }
    mask = df['property_type'] == 'Other'
    df.loc[mask, 'property_type'] = df.loc[mask, 'subtype'].map(subtype_to_type)

    # ----------------------------- #
    #  State of building encoding   #
    # ----------------------------- #
    state_order = {
        'TO REBUILD':     0,
        'TO RENOVATE':    1,
        'GOOD':           2,
        'JUST RENOVATED': 3,
        'NEW':            4,
    }
    df['Building State'] = df['building_state'].str.upper().map(state_order).fillna(2)
    df.drop(columns=['building_state'], inplace=True)

    # ----------------------------- #
    #             Region            #
    # ----------------------------- #
    df['region'] = df['zip_code'].apply(
        lambda x: 'Brussels' if 1000 <= x < 1200 else ('Flanders' if 1200 <= x < 1500 else 'Wallonia')
    )
    df.drop(columns=['zip_code'], inplace=True)

    # ----------------------------- #
    #   One-hot encoding            #
    # ----------------------------- #
    df = pd.get_dummies(df, columns=['property_type', 'region'])
    df = df.astype({col: int for col in df.select_dtypes(bool).columns})

    # S'assurer que toutes les colonnes one-hot existent
    for col in ['property_type_Apartment', 'property_type_House',
                'region_Brussels', 'region_Flanders', 'region_Wallonia']:
        if col not in df.columns:
            df[col] = 0

    # ----------------------------- #
    #   Subtype median encoding     #
    # ----------------------------- #
    encoding_map = joblib.load('preprocessing/encoding_map.pkl')
    m = encoding_map['Subtype of property']

    df['Subtype of property encoded'] = df['subtype'].str.title().map(m['map']).fillna(m['global_median'])
    df.drop(columns=['subtype'], inplace=True)

    # ----------------------------- #
    #   Distance to big cities      #
    # ----------------------------- #
    locality = df['locality'].iloc[0]
    try :
        dist = get_distances(locality)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Locality '{locality}' not found and geocoding failed.")

    for city in cities:
        df[f'dist_{city}'] = dist[f'dist_{city}']
    df['dist_sea'] = dist['dist_sea']

    df.drop(columns=['locality'], inplace=True)

    # ---------------------------------- #
    # Rename to match model + reorganize #
    # ---------------------------------- #
    df.rename(columns={
        'area':                   'Living Area',
        'rooms_number':           'Number of rooms',
        'land_area':              'Surface of the land',
        'garden':                 'Garden',
        'swimming_pool':          'Swimming pool',
        'furnished':              'Furnished',
        'terrace':                'Terrace',
        'facades_number':         'Number of facades',
        'Building State':         'state_encoded',
        'property_type_House':    'Type of property_House',
        'property_type_Apartment':'Type of property_Apartment',
        'region_Brussels':        'Region_Brussels',
        'region_Flanders':        'Region_Flanders',
        'region_Wallonia':        'Region_Wallonia',
    }, inplace=True)

    expected_cols = [
        'Number of rooms',
        'Living Area',
        'Furnished',
        'Terrace',
        'Garden',
        'Surface of the land',
        'Number of facades',
        'Swimming pool',
        'state_encoded',
        'Type of property_Apartment',
        'Type of property_House',
        'Region_Brussels',
        'Region_Flanders',
        'Region_Wallonia',
        'Subtype of property encoded',
        'dist_brussels',
        'dist_antwerp',
        'dist_ghent',
        'dist_liege',
        'dist_bruges',
        'dist_namur',
        'dist_leuven',
        'dist_sea',
    ]

    df = df[expected_cols]

    return df.to_dict(orient='records')[0]