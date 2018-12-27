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