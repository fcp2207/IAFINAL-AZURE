from abc import ABC, abstractmethod
from typing import List, Dict, Union

class LLMClientBase:
    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        # check if model exists
        cls = self.__class__
        available_models = cls.list_models()
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found. Available models are {available_models}")


    def list_models() -> List[str]:
        raise NotImplementedError
    
    @abstractmethod
    def generate_response(self, prompt: Union[List, str]):
        raise NotImplementedError
    
