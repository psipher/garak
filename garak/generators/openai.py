"""Modified OpenAI API generator for AzureOpenAI integration"""

from singleton_client import SingletonAzureClient
from typing import List, Union

# Get the singleton instance of the AzureOpenAI client
client = SingletonAzureClient.get_instance()

from garak.generators.base import Generator

class AzureOpenAIGenerator(Generator):
    def __init__(self, name, temperature=0.7, max_tokens=150, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def generate(self, prompt: Union[str, List[dict]], generations_this_call=1) -> List[str]:
        if isinstance(prompt, str):
            response = client.completions.create(
                model=self.name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop,
                n=generations_this_call
            )
            return [choice.text for choice in response.choices]

        elif isinstance(prompt, list):
            # Assuming that the structure for chat completion might need adjustment
            # based on your client's capabilities and the expected format.
            response = client.completions.create(
                model=self.name,
                messages=prompt,  # Adjust if the Azure client expects a different format
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop,
                max_tokens=self.max_tokens,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                n=generations_this_call
            )
            return [c['message']['content'] for c in response['choices']]

        else:
            raise ValueError("Unsupported input type for generation in AzureOpenAIGenerator")
        
default_class = "AzureOpenAIGenerator"
