 
from abc import ABC, abstractmethod

from .models import Action, BackendOutput, MessageType


class BackendModel(ABC):
    @abstractmethod
    def chat(self, contents: list[tuple[str, MessageType]]) -> BackendOutput: ...

    @abstractmethod
    def reset(
        self,
        system_message: str,
        action_space: list[Action] | None,
    ): ...

    @abstractmethod
    def get_token_usage(self): ...
