 
from abc import ABC, abstractmethod

from .models import Action, ActionOutput, Message


class AgentPolicy(ABC):
    @abstractmethod
    def chat(
        self,
        observation: dict[str, list[Message]],
    ) -> list[ActionOutput]: ...

    @abstractmethod
    def reset(
        self,
        task_description: str,
        action_spaces: dict[str, list[Action]],
        env_descriptions: dict[str, str],
    ) -> None: ...

    @abstractmethod
    def get_token_usage(self) -> int: ...

    @abstractmethod
    def get_backend_model_name(self) -> str: ...
