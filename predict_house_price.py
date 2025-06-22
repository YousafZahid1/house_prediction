import pandas as pd
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset

df = pd.read_csv('Housing.csv')
print(df.head())


x = df.drop(['price'], axis=1)
y = df['price']


X=pd.get_dummies(x)


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


#predicting a single value

columns = np.array(['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus'])
print(columns)

'''
To Predict A Single House Price: 
'''
data = np.array([[7420, 4, 2, 3, 'yes', 'no', 'no', 'no', 'yes', 2, 'yes', 'furnished']])
#7420,4,2,3,yes,no,no,no,yes,2,yes,furnished
data = pd.DataFrame(data, columns=columns)

sample_pred = pd.get_dummies(data)
sample_pred = sample_pred.reindex(columns=x_train.columns, fill_value=0)
store=model.predict(sample_pred)
print(store)


'''
Accuracy Score
'''

r2 = model.score(x_test, y_test)
print(f"R² Score of Linear Model is: {r2:.2f} ({r2 * 100:.2f}%)")


#1)
#convert price to categorical for classification
price = store
if price<300000:
    print("The price of the house is low")
elif price<600000:
    print("The price of the house is medium")
    
else:
    print("The price of the house is high")
sns.boxplot(x='price', y='bedrooms', data=df)
plt.show()


#2)
#plotting the predicted vs actual values

sns.pairplot(df, x_vars=['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus'], y_vars='price', height=5, aspect=0.7,hue='price', palette='coolwarm')


#3)
#Box Plot
read_data = pd.read_csv('Housing.csv')
sns.boxplot(data=read_data,x='price',y='bedrooms',hue='price',palette='coolwarm')
plt.show()

