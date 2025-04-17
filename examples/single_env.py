 
from termcolor import colored
import sys

from kgce import Benchmark, create_benchmark
from kgce.agents.backend_models import OpenAIModel
from kgce.agents.policies import SingleAgentPolicy
from kgce.benchmarks.template import template_benchmark_config
from kgce.agents.backend_models import BackendModelConfig


def start_benchmark(benchmark: Benchmark, agent: SingleAgentPolicy):
    for step in range(20):
        print("=" * 40)
        print(f"Start agent step {step}:")
        observation = benchmark.observe()["template_env"]
        print(f"Current enviornment observation: {observation}")
        response = agent.chat(
            {
                "template_env": [
                    (f"Current enviornment observation: {observation}", 0),
                ]
            }
        )
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
    benchmark = create_benchmark(template_benchmark_config)
    task, action_space = benchmark.start_task("0")
    env_descriptions = benchmark.get_env_descriptions()
    base_url = "https://api.groq.com/openai/v1"  # 获取 access_token 的 URL
    api_key = "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"  # 替换为你的 API Key
    # 创建配置对象
    model_config = BackendModelConfig(
        model_class="groq",  # 指定模型类型为 "qianfan"
        model_name="llama3-8b-8192",  # 指定具体模型名称
        base_url=base_url,  # 获取 access_token 的 URL
        api_key=api_key,  # 替换为你的 API Key
    )

    agent = SingleAgentPolicy(model_backend=model_config)
    agent.reset(task.description, action_space, env_descriptions)
    print("Start performing task: " + colored(f'"{task.description}"', "green"))
    start_benchmark(benchmark, agent)
    benchmark.reset()
