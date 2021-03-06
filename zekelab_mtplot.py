import numpy as np
arr=np.random.random((100),)

import matplotlib.pyplot as plt

#create histogram for random 100 numbers
print(plt.hist(arr, bins=5))
plt.title('Histogram')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')

#create an array with 10000 element with normal distrubition
arr1=np.random.randn((200000),)
plt.hist(arr1, bins=50)
plt.title('HistogramRdn')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')


# for equally distribution
x=np.linspace(0,10,101)
m=2
c=3
y=m*x+c
plt.plot(x,y,label='x')
plt.plot(x,y+1,label='y')
plt.legend() # it will add label as title to plot
plt.scatter(x,y,s=1)
c=np.array((y>2),dtype='int')
plt.scatter(x,y,label='line',s=1,c=c)



#plot y=x^2 from -2 to 2 and radius of point should be equal to the value and clour should be diff for value >1
x=np.linspace(-2,2,100)
y=x*x
c=np.array((y>1),dtype='int')
plt.figure(figsize=(18,7))
#fig,ax=plt.subplot(nrows=4,ncols=3,figsize=(15,3))
#plt.scatter(x,y,s=y,c=c)



# doubling the width of markers
x1 = [0,2,4,6,8,10]
y1 = [0]*len(x1)
s1 = [20*4**n for n in range(len(x1))]
plt.scatter(x1,y1,s=s1)
plt.show()

#plot for sin 2 cycles

x=-np.linspace(0,4*np.pi,200)
y=np.sin(x)
plt.plot(x,y)
plt.grid(True) # will show in grid

# plot directly for dataframe
import pandas as pd
df=pd.DataFrame({'A':range(10),'B':range(5,15)})
df.plot.line(title='plot')


from sklearn.datasets import load_iris
x=np.random.random((500,))*3
noise=np.random.randn(500)
mactual, cactual=2,3
y=mactual*x+cactual+noise*0.9
plt.scatter(x,y)
plt.plot(x,y) # do not use when SD is more


from sklearn.linear_model import LinearRegression
model=LinearRegression()
x=x.reshape(-1,1)
model.fit(x,y)
print((model.intercept_)) #bias for LP
print(model.coef_)  # coffecient


#y=x^2  from -3 to 3 and fit line with noise of SD .5 and plot it
x1=np.random.random((500))*(-3)
x2=np.random.random((500))*(3)
arr=np.append(x1,x2)
mactual1, cactual1=2,3
noise1=np.random.randn(1000)
y1=arr*arr+noise1*0.5
print(plt.scatter(arr, y1))
plt.plot(arr,y1) # do not use when SD is more