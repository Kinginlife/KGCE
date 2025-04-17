 
import networkx as nx

from kgce import BenchmarkConfig, Task, action, evaluator
from kgce.environments.template import set_state, template_environment_config


@evaluator
def is_system_state(env) -> bool:
    return env.state


@evaluator(env_name="root")
def check_submit_true(env) -> bool:
    if env.trajectory:
        action_name, params, _ = env.trajectory[-1]
        print(action_name, params)
        if action_name == "_submit" and params["content"]:
            return True
    return False


@action(env_name="root")
def _submit(content: bool) -> None:
    """Submit your answer through this function.

    Args:
        content: the content to submit
    """
    pass


template_benchmark_config = BenchmarkConfig(
    name="template_benchmark",
    environments=[template_environment_config],
    tasks=[
        Task(
            id="0",
            description="Set the system state to True.",
            evaluator=is_system_state,
            setup=set_state(False),
        ),
        Task(
            id="1",
            description="Submit True.",
            evaluator=check_submit_true,
            extra_action=[_submit],
        ),
    ],
)


@evaluator(env_name="testenv0")
def check_sys0(env) -> bool:
    return env.state


@evaluator(env_name="testenv1")
def check_sys1(env) -> bool:
    return env.state


@evaluator(env_name="testenv2")
def check_sys2(env) -> bool:
    return env.state


eval_g = nx.DiGraph()
eval_g.add_edge(check_sys0, check_submit_true)
eval_g.add_edge(check_sys1, check_submit_true)
eval_g.add_edge(check_sys2, check_submit_true)

multienv_template_benchmark_config = BenchmarkConfig(
    name="mutlienv_template_benchmark",
    environments=[
        template_environment_config.model_copy(update={"name": f"testenv{idx}"})
        for idx in range(3)
    ],
    tasks=[
        Task(
            id="0",
            description=(
                "Set the system state to True in all three environments. "
                "Then submit True to finish the project."
            ),
            evaluator=eval_g,
            extra_action=[_submit],
        )
    ],
    multienv=True,
)
