import pandas as pd

# 读取原始文件
df = pd.read_excel("/Users/xlzion/Desktop/library/RAG/graph/src/filtered_file.xlsx")



# 额外保存前1万行（无论H列是否为空）
top_10000_df = df.head(10000)  # 前1万行

# 分别保存到两个文件

top_10000_df.to_excel("top_10000_rows.xlsx", index=False)

