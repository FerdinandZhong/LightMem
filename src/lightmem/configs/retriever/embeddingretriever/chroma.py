from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Optional


class ChromaConfig(BaseModel):
    collection_name: str = Field("lightmem", description="Name of the collection")
    embedding_model_dims: Optional[int] = Field(1024, description="Dimensions of the embedding model")
    host: Optional[str] = Field(None, description="Host address for remote Chroma server")
    port: Optional[int] = Field(8000, description="Port for remote Chroma server")
    path: Optional[str] = Field("/tmp/chroma", description="Path for local persistent Chroma database")
    ssl: Optional[bool] = Field(False, description="Enable SSL for remote connections")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers for remote connections")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        extra_fields = set(values.keys()) - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. "
                f"Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = {
        "arbitrary_types_allowed": True,
    }
