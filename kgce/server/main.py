 
import os

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from kgce import EnvironmentConfig, create_environment

from .api import api_router
from .config import EnvSettings, Settings, parse_args
from .exception_handlers import (
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from .logger import LOGGING_CONFIG
from .middleware import log_request_middleware
from .utils import get_benchmarks_environments


def init(environment_config: EnvironmentConfig) -> FastAPI:
    app = FastAPI(title="Desktop Agent Benchmark Environment Server")

    app.middleware("http")(log_request_middleware)
    app.add_exception_handler(
        RequestValidationError, request_validation_exception_handler
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(api_router)

    app.environment = create_environment(environment_config)
    return app


if __name__ == "__main__":
    env_settings = EnvSettings()
    for field in env_settings.model_fields.keys():
        value = getattr(env_settings, field)
        os.environ[field] = value

    args = parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    settings = Settings(**kwargs)

    benchmarks, environments = get_benchmarks_environments()
    app = init(environment_config=environments[settings.ENVIRONMENT])

    app.server_settings = settings
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        access_log=False,
        log_config=LOGGING_CONFIG,
    )
