#!/usr/bin/env python

# Import pandas, sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

print("Running linear modelling of data python script")
print()

# Set notebook variables
if len(sys.argv) < 2:
    print("Missing filename")
    sys.exit(-1)

filename = sys.argv[1]

print("Loading filename {}".format(filename))
print()

# Use read_csv() to read regrex1.csv file
dataset = pd.read_csv(filename)
dataset.describe()
print(dataset)

# Plot data
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('Raw y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('py_orig.png')


# Fitting Linear Regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# Visualizing the Linear Regression results
# Scatter plot of original dataset
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.title('Raw y vs x')
plt.xlabel('x')
plt.ylabel('y')


# Linear model of dataset
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.title('Linear Model of y vs x')
plt.xlabel('x')
plt.ylabel('y')


# Combined Plot
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('Linear Model of y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig('py_lm.png')



