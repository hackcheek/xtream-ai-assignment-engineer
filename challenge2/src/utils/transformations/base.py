import abc

from typing import Any


class BaseDataTransform(abc.ABC):
    """
    This clase should provide preprocessing and postprocessing transformations on a specific dataset
    """
    @classmethod
    def preprocessing(cls, data: Any) -> tuple[Any, Any, Any]:
        return NotImplemented

    @classmethod
    def postprocessing(cls, current_data: Any, user_data: Any) -> Any:
        return NotImplemented
