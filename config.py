import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    render_postgres_url: Optional[str] = os.getenv("RENDER_POSTGRES_URL")
    aviationstack_api_key: Optional[str] = os.getenv("AVIATIONSTACK_API_KEY")
    default_origin_iata: str = os.getenv("DEFAULT_ORIGIN_IATA", "BOM")
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")


settings = Settings()
