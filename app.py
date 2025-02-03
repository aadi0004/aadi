from flask import Flask,jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
app = Flask(__name__)
df = pd.read_csv("covid_toy.csv")
df=df.dropna()
lb=LabelEncoder()
df['gender']=lb.fit_transform(df['gender'])
df['cough']=lb.fit_transform(df['cough'])
df['has_covid']=lb.fit_transform(df['has_covid'])
df['city']=lb.fit_transform(df['city'])
x=df.drop(columns='has_covid')
y=df['has_covid']
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
lr=LinearRegression()
lr.fit(x_train_scaled ,y_train)
y_pred=lr.predict(x_test_scaled)
from sklearn.metrics import r2_score
score=r2_score(y_pred,y_test)*100

from flask import Flask
#@app.route('/')
#def home():
 #   return "Welcome to the COVID Prediction API. Visit /accuracy to see the accuracy score."

@app.route('/')
def accuracy():
    return jsonify('Accuracy is ', score)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
