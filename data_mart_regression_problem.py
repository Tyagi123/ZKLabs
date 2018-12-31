import pandas as pd

import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
# importing linear regressionfrom sklearn
from sklearn.linear_model import LinearRegression

#import data
test_data=pd.read_csv('Assignment/Test_data_mart.csv')
train_data=pd.read_csv('Assignment/Train_data_mart.csv')
sample_date=pd.read_csv('Assignment/SampleSubmission_datamart.csv')

# Linear regression
lreg = LinearRegression()
#splitting into training and cv for cross validation
X = train_data.loc[:,['Outlet_Establishment_Year','Item_MRP']]
x_train, x_cv, y_train, y_cv = train_test_split(X,train_data.Item_Outlet_Sales)

#training the model
lreg.fit(x_train,y_train)

#predicting on cv
pred = lreg.predict(x_cv)

#calculating mse
mse = np.mean((pred - y_cv)**2)

# calculating coefficients
coeff = DataFrame(x_train.columns)
coeff['Coefficient Estimate'] = Series(lreg.coef_)

print('Coefficient -   '+str(coeff))

#Print score
print(lreg.score(x_cv, y_cv))


#residual plot
import matplotlib.pyplot as plt
x_plot = plt.scatter(pred, (pred - y_cv), c='b')
plt.hlines(y=0, xmin= -1000, xmax=5000)
plt.title('Residual plot for Linear Regression')
plt.show()

# Apply Ridge and Lasso regression
from sklearn.linear_model import Ridge,Lasso
rd=Ridge()
las=Lasso()

# fit it
rd.fit(x_train,y_train)
las.fit(x_train,y_train)

# predict output
rd_out=rd.predict(x_cv)
las_out=las.predict(x_cv)

#residual plot for Ridge regression
import matplotlib.pyplot as plt
x_plot = plt.scatter(rd_out, (rd_out - y_cv), c='b')
plt.hlines(y=0, xmin= -1000, xmax=5000)
plt.title('Residual plot for Ridge Regression')
plt.show()

#residual plot for Lasso regression
import matplotlib.pyplot as plt
x_plot = plt.scatter(las_out, (las_out - y_cv), c='b')
plt.hlines(y=0, xmin= -1000, xmax=5000)
plt.title('Residual plot for Lasso Regression')
plt.show()



from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

# mean absolute error for all regression
print('Linear regression mean absolute error = ' + str(mean_absolute_error(pred,y_cv)))
print('Ridge regression mean absolute error = ' + str(mean_absolute_error(rd_out,y_cv)))
print('Lasso regression mean absolute error = ' + str(mean_absolute_error(las_out,y_cv)))

# mean squared error for all regression
print('Linear regression mean squared error = ' + str(mean_squared_error(pred,y_cv)))
print('Ridge regression mean squared error = ' + str(mean_squared_error(rd_out,y_cv)))
print('Lasso regression mean squared error = ' + str(mean_squared_error(las_out,y_cv)))
