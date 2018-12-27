import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x1=np.random.randint(2, size=1000)
x2=np.random.randint(2, size=1000)

#create Dataframe for x1 and x2
y=x1*x2
data=pd.DataFrame({'x1':x1,'x2':x2},columns=['x1','x2'])

# import LogisticRegression and fit it
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(data,y)

x3=np.random.randint(2, size=1000)
x4=np.random.randint(2, size=1000)
datatest=pd.DataFrame({'x1':x3,'x2':x4},columns=['x1','x2'])

#create Dataframe for x3 and x4
datatest=pd.DataFrame({'x1':x3,'x2':x4},columns=['x1','x2'])
y_out_actual=x3*x4

#=create prediction for datatest
y_out=logr.predict(datatest)
print('LogisticRegression bias - %s and variance -%s' % (logr.intercept_,logr.coef_))

print(plt.scatter(data['x1'], data['x2'], color='black', s=600))
print(plt.plot(y, y_out, color='blue', linewidth=3))
plt.show()

print(plt.scatter(datatest['x1'], datatest['x2'], color='black', s=600))
print(plt.plot(y, y_out, color='blue', linewidth=3))
plt.show()

# to get classification report
from sklearn.metrics import classification_report
print(classification_report(y_out_actual, y_out))


# seaborn plots
import seaborn as sns

print(sns.countplot(x='x1', data=data))
print(sns.countplot(x='x1', hue='x2', data=data))
print(data['x1'].plot.hist())

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_out_actual, y_out))

#Accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_out_actual, y_out))