from src.odbc import selectSQLPandas

sql = "select * from ceo_data limit 10"

df = selectSQLPandas(sql=sql, uri='mysql')

rownum = 1
for _,row in df.iterrows():
    print(rownum,row['url'])
    rownum += 1