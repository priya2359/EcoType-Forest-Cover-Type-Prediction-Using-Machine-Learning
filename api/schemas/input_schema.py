# filename: api/schemas/input_schema.py
# purpose:  Pydantic v2 input/output schemas for the prediction API
# version:  1.0

from pydantic import BaseModel, Field


class CoverTypeInput(BaseModel):
    # 10 quantitative features with validated ranges
    Elevation: int = Field(ge=1859, le=3858, description="Elevation in meters")
    Aspect: int = Field(ge=0, le=360, description="Aspect in degrees")
    Slope: int = Field(ge=0, le=66, description="Slope in degrees")
    Horizontal_Distance_To_Hydrology: int = Field(ge=0, description="Horizontal dist to water (m)")
    Vertical_Distance_To_Hydrology: int = Field(description="Vertical dist to water (m)")
    Horizontal_Distance_To_Roadways: int = Field(ge=0, description="Horizontal dist to roads (m)")
    Hillshade_9am: int = Field(ge=0, le=255, description="Hillshade at 9am (0-255)")
    Hillshade_Noon: int = Field(ge=0, le=255, description="Hillshade at noon (0-255)")
    Hillshade_3pm: int = Field(ge=0, le=255, description="Hillshade at 3pm (0-255)")
    Horizontal_Distance_To_Fire_Points: int = Field(ge=0, description="Horizontal dist to fire points (m)")
    # 2 categorical inputs — API handles OHE internally
    Wilderness_Area: int = Field(ge=1, le=4, description="Wilderness area (1-4)")
    Soil_Type: int = Field(ge=1, le=40, description="Soil type (1-40)")

    model_config = {"json_schema_extra": {
        "example": {
            "Elevation": 2596, "Aspect": 51, "Slope": 3,
            "Horizontal_Distance_To_Hydrology": 258, "Vertical_Distance_To_Hydrology": 0,
            "Horizontal_Distance_To_Roadways": 510, "Hillshade_9am": 221,
            "Hillshade_Noon": 232, "Hillshade_3pm": 148,
            "Horizontal_Distance_To_Fire_Points": 6279,
            "Wilderness_Area": 1, "Soil_Type": 29
        }
    }}


class CoverTypePrediction(BaseModel):
    cover_type_id: int
    cover_type_name: str
    confidence: float
    probabilities: dict[str, float]
