from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from . import genetic_model
from django.http import HttpResponse
import matplotlib.pyplot as plt
import seaborn as sn
import io
import urllib,base64
import time



'''

-----to do----------:

- improve ga page css
- add more data to ga page
- add jupyter notebook html link
- finish with css improvements on the home and visuals page
 
'''
snsurl = ''
def serve_jupyter(request):
    return render(request, 'CustomGA.html')
def homepage(request):
    return render(request, 'home.html')

def visualpage(request):
    global snsurl
    dataset = load_boston()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    op = dataset.target.tolist()
    plots = list()
    for col in df.columns:
        subplots=list()
        for i in range(len(df[col].to_list())):
            ls = df[col].to_list()
            dct = {
                'x': ls[i],
                'y': op[i]
            }
            subplots.append(dct)
        plots.append(subplots)

    df['MEDV'] = pd.Series(dataset.target)
    #lets try seaborn
    if(snsurl == ''):
        corr_matrix = df.corr().mul(100).astype(int)
        sn.set(font_scale=0.6)
        fig = sn.heatmap(corr_matrix, annot=True)
        buf = io.BytesIO()
        fig.get_figure().savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        snsurl = urllib.parse.quote(string)

    plots = zip(plots, dataset.feature_names.tolist())
    return render(request, 'visuals.html', {'scatterplots': plots, 'url':snsurl})
def genetic_view(request):
    render_obj = {}
    dataset = load_boston()
    X, y = dataset.data, dataset.target
    params = request.session.get('params')
    linear_score = request.session.get('linear_score')
    n_gen = int(params['num_of_generations'])
    p_size = int(params['population_size'])
    m_rate = int(params['mutation_rate'])
    m_rate = float(m_rate/100)
    children = 5
    estimator = LinearRegression()
    genetic_model_obj = genetic_model.GeneticAlgorithm(estimator, n_gen, p_size, m_rate, int(p_size/children), int(p_size/children), children)
    start_time = time.time()
    genetic_model_obj.fit(X, y)
    final_score = np.mean(-1.0*cross_val_score(estimator, X[:,genetic_model_obj.chromosomes_best[-1]], y, scoring='neg_mean_squared_error'))
    end_time = time.time()
    feature_list = genetic_model_obj.get_best_features(dataset.feature_names)
    feature_list = list(map(list,feature_list))
    print(feature_list)
    render_obj['final_score'] = final_score

    render_obj['timediff'] = (end_time-start_time)*100
    render_obj['feature_list'] = feature_list
    render_obj['bardata'] = [float(linear_score), final_score]
    return render(request,'ga.html',render_obj)

def codepage(request):
    render_object = {}
    if(request.method == 'POST'):
        params = request.POST
        request.session['params'] = params
        return redirect('/ga/')
    else:
        dataset = load_boston()
        X, y = dataset.data, dataset.target
        features = dataset.feature_names
        estimator = LinearRegression()
        model = LinearRegression()
        start_time = time.time()
        model.fit(X,y)
        initial_score = np.mean(-1.0*(cross_val_score(estimator, X, y, scoring='neg_mean_squared_error')))
        end_time = time.time()
        render_object['initial_score'] = initial_score
        render_object['acc'] = 100-initial_score
        render_object['intercept'] = model.intercept_
        render_object['coefficient'] = model.coef_
        render_object['timediff'] = (end_time - start_time)*100
        request.session['linear_score'] = initial_score

    
    return render(request, 'code.html', render_object)

