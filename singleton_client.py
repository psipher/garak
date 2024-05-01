# singleton_client.py
from openai import AzureOpenAI

class SingletonAzureClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AzureOpenAI(
                api_key="###################",
                api_version="2024-02-15-preview",
                azure_endpoint="https://your_url.openai.azure.com/"
            )
        return cls._instance
