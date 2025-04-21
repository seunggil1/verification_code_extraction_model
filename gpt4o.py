from openai import OpenAI
import pandas as pd
import os

api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('sms2.csv', encoding='utf-8')
client = OpenAI(
    api_key=api_key
)

results = []
for index, row in df.iterrows():
    messages = [
        {"role": "system",
         "content": "- 문자 메세지의 내용을 읽고, 인증번호 관련 문자일 경우, 인증번호만 추출하는 AI입니다.\n- 인증번호가 있으면 해당 숫자만 답변하고, 없을경우 0으로 답변합니다."},
        {"role": "user", "content": row['Body']}
    ]

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    sms_code = result.choices[0].message.content
    print(sms_code)
    results.append(sms_code)

df['인증번호'] = results

df.to_csv('sms2.csv', encoding='utf-8', index=False)
