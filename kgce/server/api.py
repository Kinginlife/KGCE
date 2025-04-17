 
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from kgce.utils import (
    base64_to_callable,
    decrypt_message,
    encrypt_message,
    generate_key_from_env,
)

from .logger import kgce_logger as logger

api_router = APIRouter()


@api_router.post("/raw_action")
async def raw_action(request: Request):
    """Perform the specified action with given parameters."""
    enc_key = generate_key_from_env()
    # Extract query parameters as a dictionary
    request_content = await request.body()
    request_content = request_content.decode("utf-8")
    if enc_key is not None:
        request_content = decrypt_message(request_content, enc_key)
    request_json = json.loads(request_content)

    action = request_json["action"]
    parameters = request_json["parameters"]
    entry = base64_to_callable(action["dumped_entry"])
    logger.info(f"remote action: {action['name']} received. parameters: {parameters}")
    if "env" in action["kept_params"]:
        parameters["env"] = request.app.environment

    resp_data = {"action_returns": entry(**parameters)}
    if enc_key is None:
        return JSONResponse(content=resp_data)
    else:
        encrypted = encrypt_message(json.dumps(resp_data), enc_key)
        return PlainTextResponse(content=encrypted)
