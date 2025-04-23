import pandas as pd


df = pd.read_csv("sms.csv")
df['인증번호'] = df['인증번호'].apply(lambda x : int(x))

df_include_code = df[df['인증번호'] != 0]
df_exclude_code = df[df['인증번호'] == 0]


# sampling 100 rows
df_include_code_sample = df_include_code.sample(n=(df_include_code.shape[0] // 10), random_state=42)
df_include_code_rest = df_include_code.drop(df_include_code_sample.index)

df_exclude_code_sample = df_exclude_code.sample(n=(df_exclude_code.shape[0] // 10), random_state=42)
df_exclude_code_rest = df_exclude_code.drop(df_exclude_code_sample.index)


# conbine
df_train = pd.concat([df_include_code_rest, df_exclude_code_rest])
df_test = pd.concat([df_include_code_sample, df_exclude_code_sample])


# save to csv without header
df_train.to_csv("train.csv", index=False, header=False)
df_test.to_csv("test.csv", index=False, header=False)