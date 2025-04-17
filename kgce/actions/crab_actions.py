 
from time import sleep

from kgce import action, evaluator


@action(env_name="root")
def submit(content: str) -> None:
    """Submit your answer through this action. For exmaple, if you are required to
    submit a word "apple", you can use submit(content="apple").

    Args:
        content: the content to submit
    """
    pass


@evaluator(env_name="root")
def check_submit(text: str, env) -> bool:
    if env.trajectory:
        action_name, params, _ = env.trajectory[-1]
        if action_name == "submit" and text in params["content"]:
            return True
    return False


@action(env_name="root")
def complete() -> bool:
    """When you think the task is completed, use this action to notify the system. For
    exmaple, if you successfully complete the task, you can use complete().
    """
    pass


@action(env_name="root")
def wait() -> bool:
    """If the environment is still processing your action and you have nothing to do in
    this step, you can use wait().
    """
    sleep(5)


def get_element_position(element_id, env):
    """Get element position provided by function `zs_object_detection`"""
    if element_id >= len(env.element_position_map):
        raise ValueError(f"element_id {element_id} 超出范围，最大有效索引为 {len(env.element_position_map) - 1}")
    box = env.element_position_map[element_id]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return round(x), round(y)
