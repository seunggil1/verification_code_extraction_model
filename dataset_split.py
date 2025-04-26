import argparse

import pandas as pd


def main(args):
    required_column = {"Body", "인증번호"}
    df = pd.read_csv(args.input_csv_file_path, encoding="utf-8", dtype=str)

    if not required_column.issubset(df.columns):
        raise ValueError(
            f"Input CSV file must contain the following columns: {required_column}"
        )

    df_include_code = df[df["인증번호"] != "0"]
    df_exclude_code = df[df["인증번호"] == "0"]

    # sampling 100 rows
    df_include_code_sample = df_include_code.sample(
        n=(df_include_code.shape[0] // 10), random_state=42
    )
    df_include_code_rest = df_include_code.drop(df_include_code_sample.index)

    df_exclude_code_sample = df_exclude_code.sample(
        n=(df_exclude_code.shape[0] // 10), random_state=42
    )
    df_exclude_code_rest = df_exclude_code.drop(df_exclude_code_sample.index)

    # conbine
    df_train = pd.concat([df_include_code_rest, df_exclude_code_rest])
    df_test = pd.concat([df_include_code_sample, df_exclude_code_sample])

    # save to csv without header
    df_train.to_csv(f"{args.output_csv_file_dir}/train.csv", index=False, header=False)
    df_test.to_csv(f"{args.output_csv_file_dir}/test.csv", index=False, header=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Make label for dataset")
    argparser.add_argument("--input_csv_file_path", type=str, default="sms.csv")
    argparser.add_argument("--output_csv_file_dir", type=str, default=".")
    parsed_args = argparser.parse_args()
    main(parsed_args)
