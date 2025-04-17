 
import argparse

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ENVIRONMENT: str = "template_environment_config"


class EnvSettings(BaseSettings):
    DISPLAY: str = ":0"


def parse_args():
    parser = argparse.ArgumentParser(description="Application settings")
    parser.add_argument("--HOST", type=str, help="Host of the application")
    parser.add_argument("--PORT", type=int, help="Port of the application")
    parser.add_argument("--ENVIRONMENT", type=str, help="Environment to be loaded")
    return parser.parse_args()
