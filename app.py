from flask import Flask, render_template, request
from flask import redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

# Load data
df = pd.read_csv('test.csv')

# Try to load the pickle models or create new ones if there's an error
try:
    clf = pickle.load(open('out.pkl','rb'))
    elc = pickle.load(open('elct.pkl','rb'))
except (ModuleNotFoundError, ImportError):
    print("Warning: Could not load existing model pickle files.")
    print("Creating backup models for demonstration purposes.")
    # Create simple linear regression models as fallbacks
    X = np.array([[0,50,100,2,0,0,0], [2,20,20,1,1,2,1]])
    y_damage = np.array([75, 25])  # Example damage percentages
    y_outage = np.array([48, 5])   # Example outage hours
    
    # Create and train simple models
    clf = LinearRegression()
    clf.fit(X, y_damage)
    
    elc = LinearRegression()
    elc.fit(X, y_outage)


@app.route('/')
def index():
   return render_template('disaster.html')




@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      #alert = result['alert']
      wind = result['windspeed']
      rain = result['rainfall']
      sea = result['sea']
      #river = result['river']
      #dev = result['devlop']
      #dis = result['disaster']
      
      #alert = int(alert)
      wind = int(wind)
      rain = int(rain)
      sea = int(sea)
      #river = int(river)
      #dev = int(dev)
      #dis = int(dis)
      
      option1 = request.form['options1']
      alert = int(option1)
      option2 = request.form['options2']
      river = int(option2)
      option3 = request.form['options3']
      dev = int(option3)
      option4 = request.form['options4']
      dis = int(option4)
      
      y = [[alert,wind,rain,sea,river,dev,dis]]
      y = np.array(y)
          
      x = clf.predict(y) 
          
      z = elc.predict(y)
      
      if alert == 0:
          alert = 'Red'
      elif alert==1:
          alert ='Orange'
      else:
          alert = 'Yellow'
          
          
      if river == 0:
          river = 'Yes'
      else:
          river = 'No'
          
      if dev == 0:
          dev = 'Fully'
      elif dev==1:
          dev ='Semi'
      else:
          dev = 'Under'
          
      if dis == 0:
          dis = 'Cyclone'
      else:
          dis = 'Flood'
          
      context1 = {'x':x,'z':z,'alert':alert,'rain':rain,'wind':wind,'sea':sea,'river':river,'dev':dev,'dis':dis}
      #context2 = {'z':z}
      return render_template("result.html",result=result,**context1)

if __name__ == '__main__':
   app.run(debug = True)