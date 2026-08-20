from openai import OpenAI

client = OpenAI(
    api_key="gw_test-0e6_89dc08d9356db7bb025667cefed34320a6c8423ea1108e04",
    base_url="https://api.badtheorylabs.com/v1",
)

response = client.chat.completions.create(
    model="free",
    messages=[{"role": "user", "content": "which model are you using?"}],
)

print(response.choices[0].message.content)
