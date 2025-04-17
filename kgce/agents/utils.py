 
from kgce.core import Action, ActionOutput


def combine_multi_env_action_space(
    action_space: dict[str, list[Action]] | None,
) -> list[Action]:
    """Combine multi-env action space together to fit in a single agent."""
    result = []
    if action_space is None:
        return result
    for env in action_space:
        for action in action_space[env]:
            new_action = action.model_copy()
            new_action.name = new_action.name + "_in_" + env
            new_action.description = f"In {env} environment, " + new_action.description
            result.append(new_action)
    return result


def decode_combined_action(
    output_actions: list[ActionOutput],
) -> list[ActionOutput]:
    """Decode combined action output to action output with the corresponding
    environment.
    """
    result = []
    for output in output_actions:
        if "_in_" not in output.name:
            continue
        name_env = output.name.split("_in_")
        if len(name_env) != 2:
            raise RuntimeError(
                'The decoded action name should contain the splitter "_in_".'
            )
        new_output = output.model_copy()
        new_output.name = name_env[0]
        new_output.env = name_env[1]
        result.append(new_output)
    return result


def generate_action_prompt(action_space: list[Action], expand: bool = False) -> str:
    if expand:
        return "".join(
            [
                f"[**{action.name}**:\n"
                f"action description: {action.description}\n"
                f"action arguments json schema: {action.to_openai_json_schema()}\n"
                "]\n"
                for action in action_space
            ]
        )
    else:
        return "".join(
            [f"[{action.name}: {action.description}]\n" for action in action_space]
        )


def extract_text_and_code_prompts(content: str) -> tuple[list[str], list[str]]:
    r"""Extract text and code prompts from the message content.

    Returns:
        A tuple (text_list, code_list) where, text_list is a list of text and  code_list
        is a list of extracted codes both from the content.
    """
    text_prompts: list[str] = []
    code_prompts: list[str] = []

    lines = content.split("\n")
    idx = 0
    start_idx = 0
    while idx < len(lines):
        while idx < len(lines) and (not lines[idx].lstrip().startswith("```")):
            idx += 1
        text = "\n".join(lines[start_idx:idx]).strip()
        text_prompts.append(text)

        if idx >= len(lines):
            break

        # code_type = lines[idx].strip()[3:].strip()
        idx += 1
        start_idx = idx
        while not lines[idx].lstrip().startswith("```") and idx < len(lines):
            idx += 1
        if idx >= len(lines):
            break
        code = "\n".join(lines[start_idx:idx]).strip()
        code_prompts.append(code)

        idx += 1
        start_idx = idx

    return text_prompts, code_prompts
