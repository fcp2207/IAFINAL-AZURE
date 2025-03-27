from .llm_client import LLMClientBase
import ollama
import typing
from PIL.Image import Image as PILImage
from .image_utils import image_to_base64

class OllamaClient(LLMClientBase):
    def __init__(self, model_name, *args, **kwargs):
        self.ollama_client = ollama.Client()
        super().__init__(model_name, *args, **kwargs)
        self.system_instruction =kwargs.get("system_instruction", None)
        self.message_queue = []
        if self.system_instruction:
            self.message_queue.append({
                "role": "system",
                "content": self.system_instruction
            })

        self.tools = kwargs.get("tools", {})      

    def list_models():
        models_running = ollama.list()['models']
        available_models = [model["model"] for model in models_running]
        return available_models
    
    def normalize_prompt(self, prompt: typing.Union[str, typing.List]):
        """"
        This function somehow will transform the prompt into a format that the model can understand
        """
        if isinstance(prompt, str):
            return {
                "role": "user",
                "content": prompt
            }
        elif isinstance(prompt, PILImage):
            return {
                "role": "user",
                "content": "",
                "images": [image_to_base64(prompt)]
            }
        elif isinstance(prompt, list):
            images = [img for img in prompt if isinstance(img, PILImage)]
            texts = [txt for txt in prompt if isinstance(txt, str)]
            return {
                "role": "user",
                "content": "".join(texts),
                "images": [image_to_base64(img) for img in images],
            }
    
    def generate_response(self, prompt: typing.Union[str, typing.List]):
        # adding information to the context of the conversation
        # breakpoint()
        self.message_queue.append(self.normalize_prompt(prompt))
        # breakpoint()
        response = self.ollama_client.chat(
            model=self.model_name, 
            messages=self.message_queue,
            tools=self.tools.values()
        )
        
        for tool_call in response.message.tool_calls or []:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            print(f"Calling tool {tool_name} with args {tool_args}")
            tool_function = self.tools.get(tool_name, None)
            if tool_function:
                tool_response = tool_function(**tool_args)
                return tool_response
            else:
                return "Error: Tool not found"
        return response.message.content
    
            



if __name__ == "__main__":
    ollama_client = OllamaClient(model_name="llama2:latest")
    list_models = OllamaClient.list_models()
    print(list_models)
    
    # response = ollama_client.generate_response(prompt="What is Large language Models?")
    # print(response)

