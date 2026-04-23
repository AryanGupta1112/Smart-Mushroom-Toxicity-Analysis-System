from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Smart Mushroom Toxicity Analysis System"
    api_prefix: str = "/api"
    backend_root: Path = BASE_DIR
    model_artifacts_dir: Path = BASE_DIR / "saved_models"
    data_dir: Path = BASE_DIR / "data"
    sqlite_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'prediction_history.db').as_posix()}"
    default_model_name: str = "random_forest"
    inference_model_name: str = ""
    auto_train_models: bool = True
    dataset_profile: str = "mushroom"
    training_max_rows: int = 0
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
