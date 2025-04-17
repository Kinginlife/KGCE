from termcolor import colored

from kgce import Benchmark, create_benchmark
from kgce.agents.backend_models import OpenAIModel
from kgce.agents.policies import SingleAgentPolicy
from kgce.benchmarks.template import multienv_template_benchmark_config


def start_benchmark(benchmark: Benchmark, agent: SingleAgentPolicy):
    for step in range(20):
        print("=" * 40)
        print(f"Start agent step {step}:")
        observation = benchmark.observe()
        print(f"Current enviornment observation: {observation}")
        prompt = {}
        for env, obs in observation.items():
            if env == "root":
                continue
            state = obs["current_state"]
            prompt[env] = [(f"The state of {env} is {state}", 0)]
        response = agent.chat(observation=prompt)
        print(colored(f"Agent take action: {response}", "blue"))

        for action in response:
            response = benchmark.step(
                action=action.name,
                parameters=action.arguments,
                env_name=action.env,
            )
            print(
                colored(
                    f'Action "{action.name}" success, stat: '
                    f"{response.evaluation_results}",
                    "green",
                )
            )
            if response.terminated:
                print("=" * 40)
                print(
                    colored(
                        f"Task finished, result: {response.evaluation_results}", "green"
                    )
                )
                return


if __name__ == "__main__":
    benchmark = create_benchmark(multienv_template_benchmark_config)
    task, action_space = benchmark.start_task("0")
    env_descriptions = benchmark.get_env_descriptions()

    agent = SingleAgentPolicy(model_backend=OpenAIModel("gpt-4o"))
    agent.reset(task.description, action_space, env_descriptions)
    print("Start performing task: " + colored(f'"{task.description}"', "green"))
    start_benchmark(benchmark, agent)
    benchmark.reset()
