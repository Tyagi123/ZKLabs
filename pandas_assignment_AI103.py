import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep = '\t')

#Print out all columns
print('================columns============')
print(chipo.columns)
print('================columns============')



#removing space in item_name
chipo['item_name']=chipo['item_name'].map(lambda x: x.strip())
print(chipo.head())

#Count by item name
print('================Count by item name============')
print(chipo.groupby(['item_name']).sum()['quantity'])
print('================Count by item name============')

#total revenue for the dataset
c=chipo
c['item_price']=c['item_price'].map(lambda x: x.strip('$'))   # Removed $ sign
c['item_price']=c['item_price'].map(lambda x: x.strip())
c['item_price']=c['item_price'].astype(float)                  #converted to float
print('================total revenue for the dataset============')
print((c['item_price'] * c['quantity']).sum())
print('================total revenue for the dataset============')



#find out distinct item in dataset
print('================distinct item in dataset===========')
print(c['item_name'].unique().size)
print('================distinct item in dataset===========')


#Maximum sold item
abc=chipo.groupby(['item_name']).sum()
max= abc['quantity'].max()
maxsolditems=abc[abc['quantity']==max]
print('================Maximum sold item ==========')
print(maxsolditems['quantity'])
print('================Maximum sold item ==========')