import pandas as pd

path = r'output/海外.xlsx'
df = pd.read_excel(path)
df = df[:80000]
df.to_excel(r'output/海外（部分）.xlsx', index = False)