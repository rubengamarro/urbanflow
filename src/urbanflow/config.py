from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="URBANFLOW_", env_file=".env", extra="ignore")
    root: Path = Path(".")
    api_url: str = "http://localhost:8000"

    @property
    def raw(self) -> Path:
        return self.root / "data/raw"

    @property
    def processed(self) -> Path:
        return self.root / "data/processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports/latest"

    def prepare(self) -> None:
        for directory in (self.raw, self.processed, self.artifacts, self.reports):
            directory.mkdir(parents=True, exist_ok=True)
