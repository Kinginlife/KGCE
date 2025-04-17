 
from kgce.core import Environment, EnvironmentConfig, action


@action
def set_state(value: bool, env: Environment) -> None:
    """
    Set system state to the given value.

    Args:
        value (bool): The given value to set the system state.
    """
    env.state = value


@action
def current_state(env: Environment) -> bool:
    """
    Get current system state.
    """
    return env.state


template_environment_config = EnvironmentConfig(
    name="template_env",
    action_space=[set_state],
    observation_space=[current_state],
    description="A test environment",
    info=None,
    reset=set_state(False),
)
