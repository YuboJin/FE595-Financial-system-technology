from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from URL_functions import *
import matplotlib
matplotlib.use('Agg')

# Create the app object
app = Flask(__name__)
Bootstrap(app)

# Create the route for the home page where users can see
# the information of the application.
@app.route('/')
def home():
        return render_template('home.html')

# Create the route for the input page where users can
# input parameters.
@app.route('/strategy')
def input():
        return render_template('strategy.html')

# Create the route for the results page that would show the
# results to the user.
@app.route('/results', methods=['POST'])
def predict():
        
        if request.method == 'POST':
                # Get the input from the strategy.html
                a = request.form
                etf = request.form.getlist("asset")
                s1 = request.form["strategy1"][:-1]
                beta1 = str_to_num(request.form["beta1"])
                return1 = str_to_num(request.form["return1"])
                s2 = request.form["strategy2"][:-1]
                beta2 = str_to_num(request.form["beta2"])
                return2 = str_to_num(request.form["return2"])
                train_start = request.form["train_start"]
                train_end = request.form["train_end"]
                test_start = request.form["test_start"]
                test_end = request.form["test_end"]

                # Retrieve data from the csv file
                data = pd.read_csv('data/data.csv', index_col=0)
                benchmark = data.SPY
                data = data[list(data.columns[:4])+etf]
                
                # Use the model to train and test the strategies
                w1, r_train1, r_test1 = Model(data, s1, train_start, train_end, test_start, test_end, beta_T=beta1, R_T=return1)
                w2, r_train2, r_test2 = Model(data, s2, train_start, train_end, test_start, test_end, beta_T=beta2, R_T=return2)

                # Show the optimiaed weights of assets and strategy performance
                # for the strategies
                if s1 != '' and s2 != '':
                        weight = show_weights([w1,w2], etf, [s1, s2])
                        train_fig = plot_PnLs([r_train1, r_train2], benchmark)
                        test_fig = plot_PnLs([r_test1, r_test2], benchmark)
                elif s1 != '' and s2 == '':
                        weight = show_weights([w1], etf, [s1])
                        train_fig = plot_PnLs(r_train1, benchmark)
                        test_fig = plot_PnLs(r_test1, benchmark)
                elif s1 == '' and s2 != '':
                        weight = show_weights([w2], etf, [s2])
                        train_fig = plot_PnLs(r_train2, benchmark)
                        test_fig = plot_PnLs(r_test2, benchmark)
                else:
                        pass
                  
        return render_template('results.html', weight=weight, train_fig=train_fig, test_fig=test_fig, title='Results')

if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True)
