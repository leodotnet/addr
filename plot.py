from numpy import *
import math
import matplotlib.pyplot as plt

t = linspace(0,100,10)
a = [0,
0,
0,
12.38,
51.87,
71.87,
75.28,
77.15,
79.98,
80.99]

b = [0,
0,
0,
8.47,
18.38,
57.44,
71.63,
76.33,
78.23,
79.98]


plt.plot(t, a, label='w pretrain')
plt.plot(t, b, label='w/o pretrain (dashed line)', dashes=[30, 5, 10, 5])
plt.legend()
#plt.show()
fig = plt.gcf()
fig.savefig('pretrain.eps', format='eps', dpi=900)

