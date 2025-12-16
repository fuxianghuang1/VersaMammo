import pandas as pd

# 文件路径
path_vindr = '/mnt/data/hfx/features/high/vindr/versamammo/features.pkl'
path_embed = '/mnt/data/hfx/features/high/embed/versamammo/features.pkl'
path_rsna = '/mnt/data/hfx/features/high/rsna/versamammo/features.pkl'
data_path = '/home/fuxianghuang/code/downstream_tasks/csv_files/predata.csv'

# 读取.pkl文件
df_vindr = pd.read_pickle(path_vindr)
df_embed = pd.read_pickle(path_embed)
df_rsna = pd.read_pickle(path_rsna)

# 读取.csv文件
data_df = pd.read_csv(data_path)

# 合并特征DataFrame（注意这里使用列表作为concat的输入）
featuer_df = pd.concat([df_vindr, df_embed, df_rsna], ignore_index=True)

# 所有特征DataFrame都有一个名为"image_path"的列，用于与data_df合并
all_df = pd.merge(data_df, featuer_df, on="image_path", how="inner")  # 使用inner join，也可以根据需要选择outer, left, right
all_df = all_df[['feature', 'birads', 'split']]
all_df = all_df.dropna(subset=['feature', 'birads']) 
all_df = all_df[~(all_df['birads'].isin([6]))]

# 打印 data_df 的列名
print("Columns of data_df:", data_df.columns)  
# 打印 featuer_df 的列名
print("Columns of featuer_df:", featuer_df.columns)  
# 打印合并后的DataFrame的前几行
print("First few rows of all_df:")
# print(all_df.head())
print(f"Total number of rows in all_df: {all_df.shape[0]}")  

# 统计每个 BI-RADS 类别出现的次数
birads_counts = all_df['birads'].value_counts()
print("BI-RADS categories and their counts:")
print(birads_counts)

# 分离数据库和查询集
database_df = all_df[all_df['split'] != "test"]
query_df = all_df[all_df['split'] == "test"]

print("First few rows of database_df:")
# print(database_df.head())
print(f"Total number of rows in database_df: {database_df.shape[0]}") 
print("First few rows of query_df:")
# print(query_df.head())
print(f"Total number of rows in query_df: {query_df.shape[0]}") 

# 保存为.pkl文件
database_df.to_pickle('/home/fuxianghuang/code/downstream_tasks/csv_files/features4retrieval/high/versamammo_database.pkl')
query_df.to_pickle('/home/fuxianghuang/code/downstream_tasks/csv_files/features4retrieval/high/versamammo_query.pkl')

'''
birads
1.0    335886
2.0     82318
0.0     68825
3.0     25796
4.0     11424
5.0      1426
6.0      1327
'''