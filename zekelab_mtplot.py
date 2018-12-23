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