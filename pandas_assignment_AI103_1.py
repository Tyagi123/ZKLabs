import pandas as pd
users = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user',
                      sep='|', index_col='user_id')

print(users)


#number of engineers above 50 age
print(users[users['age'] > 50])

#gender count for Each occupation
b=users.groupby(['occupation']).count()['gender']
print(b)
print(users.groupby(['occupation', 'gender']).count().iloc[:, :3])


#most popular profession for age group of 18 to 30 age bracket
data=users[(users['age']>18) & (users['age']<30)]
db=data.groupby('occupation')
maxcount=db['age'].count()
print(maxcount)
print(maxcount.max())