from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ListingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neighbourhood_group: str = Field(min_length=1, max_length=80)
    neighbourhood: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    price: float = Field(ge=0)
    minimum_nights: int = Field(ge=0)
    availability_365: int = Field(ge=0, le=365)
    calculated_host_listings_count: int = Field(ge=0)
    number_of_reviews: int | None = Field(default=None, ge=0)
    reviews_per_month: float | None = Field(default=None, ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["without_reviews", "with_reviews"]
    listing: ListingFeatures

    @model_validator(mode="after")
    def validate_review_features(self):
        if self.mode == "with_reviews":
            if self.listing.number_of_reviews is None:
                raise ValueError("number_of_reviews is required when mode='with_reviews'.")
            if self.listing.reviews_per_month is None and self.listing.number_of_reviews > 0:
                raise ValueError(
                    "reviews_per_month is required when number_of_reviews is greater than zero."
                )
        return self


class PredictionResponse(BaseModel):
    predicted_room_type: str
    predicted_class_probability: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    mode: Literal["without_reviews", "with_reviews"]
    model_version: str
    disclaimer: str
