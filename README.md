# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 


## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: ADITHYA M

RegisterNumber: 212224230008

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```



## Output:

<img width="1035" height="161" alt="Screenshot 2025-09-27 102447" src="https://github.com/user-attachments/assets/2c711633-3948-47e6-8db2-1365aba48d4c" />

<img width="735" height="732" alt="Screenshot 2025-09-27 102502" src="https://github.com/user-attachments/assets/3d85e31b-eaf5-4ada-a313-b744fee3de60" />

<img width="843" height="221" alt="Screenshot 2025-09-27 102509" src="https://github.com/user-attachments/assets/c42018b6-ec1a-4a1b-9143-537eff87defc" />

<img width="803" height="522" alt="Screenshot 2025-09-27 102516" src="https://github.com/user-attachments/assets/d3806299-d440-4eee-a5e9-3578ab88207b" />

<img width="863" height="515" alt="Screenshot 2025-09-27 102527" src="https://github.com/user-attachments/assets/7780dab6-fdc8-418c-bd55-c726935a0b11" />

<img width="599" height="175" alt="Screenshot 2025-09-27 102537" src="https://github.com/user-attachments/assets/13facb73-12de-454e-8767-35163389bc04" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
