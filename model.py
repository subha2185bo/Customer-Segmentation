#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
# importing libraries
#import os
#import numpy as np
#import flask
#import pickle
#from flask import Flask, render_template, request

"""


# In[ ]:


import os
import numpy as np
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template


# creating instance of the class
app = Flask(__name__, template_folder='templates')

# to tell flask what url should trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    
    
# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,2)
    loaded_model = pickle.load(open("cust_knn.pkl","rb")) # load the model
    result = loaded_model.predict(to_predict) # predict the values using loded model
    return result[0]


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.values()
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
            
        if float(result) == 0:
            prediction='Customer Class: Standard, who can be approached and can be a returning customer'
        elif float(result) == 1:
            prediction='Customer Class: Careless, who are impulsive in any situation'
        elif float(result) == 2:
            prediction='Customer Class: Target, the MOST PREFFERED group and returning customers'
        elif float(result) == 3:
            prediction='Customer Class: Flying, small or average pocket group'
        elif float(result) == 4:
            prediction='Customer Class: Seasonal, waits for Mega Sale Offers to spend'
            
        return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=False) # use debug = False for jupyter notebook


# <html> 
# <body> 
#     <h3>Customer Segmentation</h3> 
#   
# <div> 
#   <form action="/result2" method="POST"> 
#     <label for="Annual Income">Annual Income</label> 
#     <input type="text" id="Annual Income" name="Annual Income"> 
#     <br>
#     <label for="Spending Score (1-100)">Spending Score (1-100)</label> 
#     <input type="text" id="Spending Score (1-100)" name="Spending Score (1-100)"> 
#     <br>       
#     <input type="submit" value="Submit"> 
#   </form> 
# </div> 
# </body> 
# </html> 

# In[ ]:




