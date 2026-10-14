from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "openai": any OpenAI-compatible gateway (AnyModel by default, OpenRouter works too)
    # "claude_cli": Claude through the local Claude Code CLI (`claude -p`), no API key needed
    llm_provider: str = "openai"
    claude_cli_model: str = "haiku"
    llm_base_url: str = "https://anymodel.org/v1"
    llm_api_key: str = ""
    llm_model: str = "ag/gemini-3-flash"
    llm_daily_call_limit: int = 400

    database_url: str = "sqlite:///data/games.db"

    run_interval_minutes: int = 60
    games_per_run: int = 20
    detail_workers: int = 4
    scheduler_enabled: bool = True
    # "Every new day" is counted in the employer's timezone
    timezone: str = "Europe/Moscow"

    letsplay_enabled: bool = True
    letsplay_workers: int = 1
    letsplay_audio_minutes: int = 8
    # Re-search let's plays for games where nothing was found after this many days
    letsplay_retry_days: int = 3
    # Ways to get the blogger's speech as text, tried in this order:
    # captions (YouTube speech track), audio (cut audio -> LLM speech-to-text),
    # gemini_video (YouTube link straight to Gemini; OpenRouter only)
    letsplay_strategies: str = "captions,audio"

    # Optional: when set, POST /api/runs requires header X-Admin-Token
    admin_token: str = ""


settings = Settings()
