# coding=utf-8

from openai import OpenAI

base_url = "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/8a062fd4-7367-4ab4-a936-5eeb8fb821c4/v1" # API地址
api_key = "9fgeAVRzFQVHioMn5CL6QtR5qjh2pZcklyNz6OIaC-Mx1sU0pin3kjiurpo7WOaAuqfGKG5q4p0zccxXEA4yQA" # 把yourApiKey替换成已获取的API Key

client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
    model = "DeepSeek-R1", # 模型名称
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好"},
    ],
    max_tokens = 1024,
    temperature = 1,
    stream = False
)

print(response.choices[0].message.content)