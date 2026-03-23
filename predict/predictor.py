from fastapi import APIRouter
from pydantic import BaseModel, Field
import pandas as pd
import joblib

router = APIRouter()

class PredictionInput(BaseModel):
    number_of_rooms:             int   = Field(..., alias='Number of rooms')
    living_area:                 int   = Field(..., alias='Living Area')
    furnished:                   bool  = Field(..., alias='Furnished')
    terrace:                     bool  = Field(..., alias='Terrace')
    garden:                      bool  = Field(..., alias='Garden')
    surface_of_the_land:         int   = Field(..., alias='Surface of the land')
    number_of_facades:           int   = Field(..., alias='Number of facades')
    swimming_pool:               bool  = Field(..., alias='Swimming pool')
    state_encoded:               int   = Field(..., alias='state_encoded')
    type_of_property_apartment:  int   = Field(..., alias='Type of property_Apartment')
    type_of_property_house:      int   = Field(..., alias='Type of property_House')
    region_brussels:             int   = Field(..., alias='Region_Brussels')
    region_flanders:             int   = Field(..., alias='Region_Flanders')
    region_wallonia:             int   = Field(..., alias='Region_Wallonia')
    subtype_encoded:             float = Field(..., alias='Subtype of property encoded')
    dist_brussels:               float = Field(..., alias='dist_brussels')
    dist_antwerp:                float = Field(..., alias='dist_antwerp')
    dist_ghent:                  float = Field(..., alias='dist_ghent')
    dist_liege:                  float = Field(..., alias='dist_liege')
    dist_bruges:                 float = Field(..., alias='dist_bruges')
    dist_namur:                  float = Field(..., alias='dist_namur')
    dist_leuven:                 float = Field(..., alias='dist_leuven')

    model_config = {'populate_by_name': True}

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump(by_alias=True)])
    
@router.post("", tags=["Predictor"])
def predict(property: PredictionInput):
    models = {
        'xgboost': joblib.load('models/xgboost_model.pkl'),
        'random_forest': joblib.load('models/random_forest_model.pkl'),
        'lightgbm': joblib.load('models/lightgbm_model.pkl'),
        'catboost': joblib.load('models/catboost_model.pkl')}
    
    df = property.to_df()
    predictions = []

    for name, model in models.items():
        prediction = model.predict(df)[0]
        predictions.append(prediction)
    # mean
    result = sum(predictions) / len(predictions)
    
    return {"predicted_price": float(result)}