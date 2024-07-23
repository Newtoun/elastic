import pandas as pd

df = pd.read_csv("amazon_products.csv")
colunas = list(df.columns.values)

print(colunas)
columns_to_drop = ['imgUrl', 'productURL','asin','listPrice','isBestSeller']
df.drop(columns_to_drop, axis=1, inplace=True)

colunas = list(df.columns.values)
print(colunas)
df.drop(df.tail(1300000).index, inplace=True)
df.to_csv("amazon_productsV2.csv", index=False)
