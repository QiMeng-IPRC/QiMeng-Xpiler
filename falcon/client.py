import os

from openai import AzureOpenAI

# 从环境变量获取配置
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not endpoint or not api_key:
    raise ValueError(
        "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment."
    )

# 全局 client
client = AzureOpenAI(
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=endpoint,
)


def invoke_llm(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment,
    )
    return response.choices[0].message.content
