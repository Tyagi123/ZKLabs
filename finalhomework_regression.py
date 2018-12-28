import pandas as pd
import numpy as np
import seaborn as sen

#  create test data (0.5 std)
x1=np.append((np.random.random(500)*-3),(np.random.random(500)*3))
x2=np.append((np.random.random(500)*-3),(np.random.random(500)*3))
noise=np.random.randn(1000)
y=2*x1+3*x2+noise*0.5
data=pd.DataFrame({'x1':x1,'x2':x2,'y':y},columns=['x1','x2','y'])

#  create train data for y=2x1+3x2+noise(0.5 std)
x3=np.append((np.random.random(500)*-3),(np.random.random(500)*3))
x4=np.append((np.random.random(500)*-3),(np.random.random(500)*3))
y_actual=2*x1+3*x2+noise*0.5
datatest=pd.DataFrame({'x1':x3,'x2':x4,'y':y_actual},columns=['x1','x2','y'])


# aplly linear/lasso/ridge regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso
lr=LinearRegression()
rd=Ridge()
las=Lasso()


# fit train data for all regression
lr.fit(data[['x1','x2']],data[['y']])
rd.fit(data[['x1','x2']],data[['y']])
las.fit(data[['x1','x2']],data[['y']])

#predict test data
lr_out=lr.predict(datatest[['x1','x2']])
rd_out=rd.predict(datatest[['x1','x2']])
las_out=las.predict(datatest[['x1','x2']])



print("Linear Regression bias and cofe"+str(lr.coef_)+str(lr.intercept_))
print("Ridge Regression bias and cofe"+str(rd.coef_)+str(rd.intercept_))
print("Lasso Regression bias and cofe"+str(las.coef_)+str(las.intercept_))


import matplotlib.pyplot as plt
plt.scatter(x2,x3,color='black')
plt.plot(y_actual,lr_out)
plt.show()


plt.scatter(x2,x3,color='black')
plt.plot(rd_out,y_actual)
plt.show()


plt.scatter(datatest,data,color='black')
plt.plot(y_actual,las_out,color='red')
plt.show()




# The mean squared error
print("Linear Regression Mean squared error: %.2f" % np.mean((lr.predict(datatest[['x1','x2']]) - lr_out) ** 2))
# Explained variance score: 1 is perfect prediction
print('Linear Regression Variance score: %.2f' % lr.score(datatest[['x1','x2']], lr_out))


# The mean squared error
print("Ridge Regression Mean squared error: %.2f" % np.mean((rd.predict(datatest[['x1','x2']]) - rd_out) ** 2))
# Explained variance score: 1 is perfect prediction
print('Ridge Regression Variance score: %.2f' % rd.score(datatest[['x1','x2']], rd_out))


# The mean squared error
print("Lasso Regression Mean squared error: %.2f" % np.mean((las.predict(datatest[['x1','x2']]) - las_out) ** 2))
# Explained variance score: 1 is perfect prediction
print('Lasso Regression Variance score: %.2f' % las.score(datatest[['x1','x2']], las_out))