import argparse
import os

import pandas as pd
from openai import OpenAI


def main(args):
    api_key = os.environ["OPENAI_API_KEY"]
    required_column = {"Body"}

    df = pd.read_csv(args.input_csv_file_path, encoding="utf-8")

    if not required_column.issubset(df.columns):
        raise ValueError(
            f"Input CSV file must contain the following columns: {required_column}"
        )

    client = OpenAI(api_key=api_key)

    results = []
    for index, row in df.iterrows():
        messages = [
            {
                "role": "system",
                "content": "- 문자 메세지의 내용을 읽고, 인증번호 관련 문자일 경우, 인증번호만 추출하는 AI입니다.\n- 인증번호가 있으면 해당 숫자만 답변하고, 없을경우 0으로 답변합니다.",
            },
            {"role": "user", "content": row["Body"]},
        ]

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        sms_code = result.choices[0].message.content
        print(sms_code)
        results.append(sms_code)

    df = df.drop(columns=["인증번호"])
    df["인증번호"] = results

    df.to_csv(args.output_csv_file_path, encoding="utf-8", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Make label for dataset")
    argparser.add_argument("--input_csv_file_path", type=str, default="sms.csv")
    argparser.add_argument("--output_csv_file_path", type=str, default="sms.csv")
    parsed_args = argparser.parse_args()
    main(parsed_args)
