 
import logging
import json
import os
from kgce import Action, ActionOutput
from kgce.agents.backend_models import BackendModelConfig, create_backend_model
from kgce.agents.utils import (
    combine_multi_env_action_space,
    decode_combined_action,
    generate_action_prompt,
)
from kgce.core.agent_policy import AgentPolicy
from kgce.core.backend_model import (
    MessageType,
)
from kgce.utils.measure import timed

logger = logging.getLogger(__name__)


class SingleAgentPolicy(AgentPolicy):
    _system_prompt_with_function_call = """\
    You are a helpful assistant. Now you have to do a task as described below: 

    **"{task_description}."**

    You should never forget this task and always perform actions to achieve this task. 
    And this is the description of each given environment: {env_description}. A
    unit operation you can perform is called Action. You have a limited action space as
    function calls:
    {action_descriptions}
    You may receive a screenshot of the current system. You may receive a screenshot of
    a smartphone app. The interactive UI elements on the screenshot are labeled with
    numeric tags starting from 1. 

    In each step, You MUST explain what do you see from the current observation and the
    plan of the next action, then use a provided action in each step to achieve the
    task. You should state what action to take and what the parameters should be. Your
    answer MUST be a least one function call. You SHOULD NEVER ask me to do anything for
    you. Always do them by yourself using function calls.
    """

    _system_prompt_no_function_call = """\
    You are a helpful assistant. Now you have to do a task as described below: 

    **"{task_description}."**

    You should never forget this task and always perform actions to achieve this task. 
    And this is the description of each given environment: {env_description}. You will
    receive screenshots of the environments. The interactive UI elements on the
    screenshot are labeled with numeric tags starting from 1. 

    A unit operation you can perform is called Action. You have a limited action space
    as function calls: {action_descriptions}. You should generate JSON code blocks to
    execute the actions. Each code block MUST contains only one json object, i.e. one
    action. You can output multiple code blocks to execute multiple actions in a single
    step. You must follow the JSON format below to output the action. 
    ```json
    {{"name": "action_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
    ```
    or if not arguments needed:
    ```json
    {{"name": "action_name", "arguments": {{}}}}
    ```
    You MUST use exactly the same "action_name" as I gave to you in the action space.
    You SHOULDN'T add any comments in the code blocks.

    In each step, You MUST explain what do you see from the current observation and the
    plan of the next action, then use a provided action in each step to achieve the
    task. You should state what action to take and what the parameters should be. Your
    answer MUST contain at least one code block. You SHOULD NEVER ask me to do anything
    for you. Always do them by yourself.
    """

    def __init__(
        self,
        model_backend: BackendModelConfig,
        function_call: bool = True,
    ):
        self.model_backend = create_backend_model(model_backend)
        self.function_call = function_call
        if not self.model_backend.support_tool_call and self.function_call:
            logger.warning(
                "The backend model does not support tool call: {}".format(
                    model_backend.model_name
                )
                + "\nFallback to no function call mode."
            )
            self.function_call = False
        if self.function_call:
            self.system_prompt = self._system_prompt_with_function_call
        else:
            self.system_prompt = self._system_prompt_no_function_call
        self.reset(task_description="", action_spaces=None, env_descriptions={})
        self.knowledge_base = {}

    # Load a software page knowledge base
    def load_knowledge_base(self):
        knowledge_base = {}
        # Set the knowledge base directory path (note Windows path escaping)
        knowledge_base_dir = r"D:\桌面\多智能体\kgce\kgce-benchmark-v0\knowledge_base"

        # Assume a task description string exists (adjust the acquisition method as needed)
        task_description = self.task_description  # or pass through parameters
        # print("task description:", task_description)
        # Iterate over the knowledge base directory [1,2](@ref)
        for filename in os.listdir(knowledge_base_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(knowledge_base_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Get the first key (software name) [3](@ref)
                        software_name = next(iter(data.keys()))

                        # Conditional check: the software name must appear in the task description [4](@ref)
                        if software_name in task_description:
                            knowledge_base[software_name] = data[software_name]

                except json.JSONDecodeError:
                    print(f"[Error] JSON Decode Fail: {filename} (perhaps the file format is incorrect)")
                except KeyError:
                    print(f"[Error] {filename} lacks a valid data structure")
                except Exception as e:
                    print(f"[Exception] An unexpected error occurred while processing file {filename}: {str(e)}")
        return knowledge_base

    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> list:
        print("with knowledge base!")
        self.task_description = task_description
        self.action_space = combine_multi_env_action_space(action_spaces)
        system_message = self.system_prompt.format(
            task_description=task_description,
            action_descriptions=generate_action_prompt(
                self.action_space,
                expand=not self.function_call,
            ),
            env_description=str(env_descriptions),
        )
        # print("task_description:", task_description)
        if self.function_call:
            self.model_backend.reset(system_message, self.action_space)
        else:
            self.model_backend.reset(system_message, None)

    def get_token_usage(self):
        return self.model_backend.get_token_usage()

    def get_backend_model_name(self):
        return self.model_backend.__class__.__name__ + "_" + self.model_backend.model

    @timed
    def chat(
            self,
            observation: dict[str, list[tuple[str, MessageType]]],
    ) -> list[ActionOutput]:

        self.knowledge_base=self.load_knowledge_base()
        # 生成结构化知识提示
        knowledge_prompt = []

        # 遍历知识库结构
        for app_name, app_data in self.knowledge_base.items():
            # 应用级描述
            app_header = f"🏫【{app_name}】功能导航指南：\n"
            knowledge_prompt.append((app_header, MessageType.TEXT))

            for page_name, page_data in app_data.items():
                # 页面级描述
                page_desc = f"📄 页面：{page_name}\n   - {page_data['description']}\n"

                # 元素级描述
                elements_desc = []
                for elem_name, elem_data in page_data['elements'].items():
                    elem_entry = f"   🔘 {elem_name}：\n"
                    elem_entry += f"      📍 位置：{elem_data['position']}\n"

                    if 'description' in elem_data:
                        elem_entry += f"      📝 功能：{elem_data['description']}\n"

                    if 'sub_elements' in elem_data:
                        sub_elems = "\n".join(
                            [f"        ▪ {sub_name}：{sub_desc}"
                             for sub_name, sub_desc in elem_data['sub_elements'].items()]
                        )
                        elem_entry += f"      🎚 子功能：\n{sub_elems}\n"

                    elements_desc.append(elem_entry)

                # 组合页面信息
                knowledge_prompt.append((
                    f"{page_desc}{''.join(elements_desc)}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    MessageType.TEXT
                ))

        # 构建操作指引
        operation_guidance = [
            ("\n📌 操作路径建议：", MessageType.TEXT),
            ("1. 确认当前所在页面（通过导航栏或页面特征元素）", MessageType.TEXT),
            ("2. 根据任务目标选择功能入口（匹配元素描述和位置）", MessageType.TEXT),
            ("3. 按顺序完成子页面操作（注意弹窗和状态变更）", MessageType.TEXT),
            ("4. 使用导航栏返回或切换功能模块\n", MessageType.TEXT),
            ("⚠️ 特别注意：", MessageType.TEXT),
            ("- 下拉菜单操作需先定位再选择选项", MessageType.TEXT),
            ("- 弹窗操作后需等待页面刷新", MessageType.TEXT),
            ("- 跨页面操作需保持任务连续性\n", MessageType.TEXT)
        ]

        # 构建完整提示
        prompt = []
        prompt.extend(knowledge_prompt)
        for env in observation:
            prompt.extend(observation[env])
        prompt.append((
            f"📋 当前任务：{self.task_description}\n"
            "请按照以下步骤执行操作：\n"
            "1. 分析当前页面内容\n"
            "2. 匹配知识库中的页面特征\n"
            "3. 生成具体操作序列",
            MessageType.TEXT
        ))
        prompt.extend(operation_guidance)
        
        # 获取模型响应
        output = self.model_backend.chat(prompt)
        # print("single_chat获取到的output:",output)
        # 调试输出
        # print(f"⚙️ 生成动作：{output.action_list}")

        return decode_combined_action(output.action_list)
