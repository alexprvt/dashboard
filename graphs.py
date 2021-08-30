# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:09:14 2021

@author: Alexandre
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def default_gauge(default_proba):
    
    if default_proba <= 20:
        color = 'green'
    elif default_proba <= 40:
        color = 'green'
    elif default_proba <= 60:
        color = 'yellow'
    elif default_proba < 80:
        color = 'orange'
    else:
        color = 'red'
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = default_proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilité de défaut de paiement (%)"},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': color}}))
    
    return fig


def plotly_waterfall(sk_id, shap_series, n_feats=10, shap_modif=np.array([])):
    
    n_var = 43

    if shap_modif.any() == False:
        shap_series = pd.DataFrame(shap_series)
        shap_series = shap_series.reindex(shap_series[str(sk_id)].abs().sort_values().index)
        shap_series = -shap_series[str(sk_id)]
        values = list(shap_series.values*100)
        
    else:
        shap_series = pd.DataFrame(shap_modif, shap_series.index)
        shap_series = -shap_series.reindex(shap_series[0].abs().sort_values().index)
        values = list(shap_series.values.reshape(1, -1)[0]*100)
    
    first_val = sum(values[:(n_var-n_feats)])
    values = [first_val] + values[n_var-n_feats:]
    text = [ '%.2f' % elem for elem in values ]
    
    for i,txt in enumerate(text):
        if float(txt) > 0:
            text[i] = '+' + txt
            
    text = [txt + ' %' for txt in text]
    ticks = shap_series.index
    ticks = ['Autres'] + list(ticks[n_var-n_feats:])
    fig = go.Figure(go.Waterfall(
        orientation='h',
        x = values,
        y= ticks, base = 50,
        text = text,
        measure = ["relative" for k in range(len(text)+1)],
        increasing = {"marker":{"color":"red", "line":{"color":"#C10000", "width":2}}},
        decreasing = {"marker":{"color":"green", "line":{"color":"#2B9B00", "width":2}}},
    ))

    fig.update_layout(title = "Influence de chaque varibale sur la prédiction de difficulté de paiement", waterfallgap = 0.2)
    fig.update_layout(height=int(n_feats*50))
    fig.update_yaxes(tickangle=-30)
    VARS = ticks[1:]
    VARS.reverse()
    
    return fig, VARS


def plot_bars(df, var, sk_id, bins=np.array([]), n_bins=10):
    
    value = df.loc[sk_id][var]
    
    if np.isnan(value):
        value =  'Non renseigné'
        df2 = df.copy()
        df2[var] = df2[var].fillna('Non renseigné')
        df2 = df2[df2[var]=='Non renseigné']
        default_nan = 100*df2[df2['TARGET']==1].shape[0]/df2.shape[0]
        
        var_data = df[['TARGET', var]]
        min_var, max_var = round(df[var].min(),1), round(df[var].max(),1)
    
        if bins.any()==False:
            bins = np.linspace(min_var, max_var, num = n_bins)

        var_data['var_bin'] = pd.cut(var_data[var], bins = bins)
        var_groups  = var_data.groupby('var_bin').mean()


        y = list(100*var_groups['TARGET'])
        y.append(default_nan)
    
        x = var_groups.index.astype(str)
        x = [string.replace(",", " -") for string in x]
        x = [string.replace("(", "") for string in x]
        x = [string.replace("]", "") for string in x]
        x.append('Non renseigné')
    
        colors = ['red' for i in range(len(y))]
        colors[-1] = 'purple'
    
        fig = go.Figure([go.Bar(x=x, y=y, marker_color=colors)])
    
        fig.add_annotation(x = x[-1],
                          y = y[-1],
                          text = f"{var} du client: {value}",
                          xref = "x",
                          yref = "y",
                          showarrow = True,
                          arrowcolor='purple',
                          ax = x[-1],
                          ay = max(y)+3,
                          axref="x",
                          ayref='y',
                          )
        
        fig.add_hline(y=8.1, line_width=3, line_dash="dash", line_color="orange", annotation_text="Moyenne")
            
        fig.update_annotations(arrowcolor='purple', arrowhead=2, arrowwidth=2)
        fig.update_layout(title=f'Taux de difficulté de paiement (%) par {var}')

        return fig

    else:
        value = round(df.loc[sk_id][var],2)
    
        var_data = df[['TARGET', var]]
        min_var, max_var = round(df[var].min(),1), round(df[var].max(),1)
    
        if bins.any()==False:
            bins = np.linspace(min_var, max_var, num = n_bins)

        var_data['var_bin'] = pd.cut(var_data[var], bins = bins, include_lowest=True)
        var_groups  = var_data.groupby('var_bin').mean()


        y = 100 * var_groups['TARGET']

        x = var_groups.index.astype(str)
        x = [string.replace(",", " -") for string in x]
        x = [string.replace("(", "") for string in x]
        x = [string.replace("]", "") for string in x]

        colors = ['red' for i in range(len(y))]
        i=0
        while value not in var_groups.index[i]:
            i+=1
        colors[i] = 'purple'

        fig = go.Figure([go.Bar(x=x, y=y, marker_color=colors)])

        fig.add_annotation(x = x[i],
                      y = y[i],
                      text = f"{var} du client: {value}",
                      xref = "x",
                      yref = "y",
                      showarrow = True,
                      arrowcolor='purple',
                      ax = x[i],
                      ay = max(y)+3,
                      axref="x",
                      ayref='y',
                      )
        
        fig.add_hline(y=8.1, line_width=3, line_dash="dash", line_color="orange", annotation_text="Moyenne")
        
        fig.update_annotations(arrowcolor='purple', arrowhead=2, arrowwidth=2)
        fig.update_layout(title=f'Taux de difficulté de paiement (%) par {var}')

        return fig


def plot_categ_bars(df, var, sk_id):

    default_tab = df[df.TARGET==1].groupby(var)['TARGET'].count()
    total_tab = df.groupby(var)['TARGET'].count()

    ind = total_tab.index
    default = []
    
    for i, index in enumerate(total_tab.index):
        if index in default_tab.index:
            default.append(default_tab[index]/total_tab[index]*100)
        else:
            default.append(0)
     
    colors=['red' for k in range(len(default))]
    
    value = df.loc[sk_id][var]
    i=0
    while value != total_tab.index[i]:
        i+=1
    colors[i] = 'purple'

    fig = go.Figure([go.Bar(x=ind, y=default, marker_color=colors)])
    fig.add_hline(y=8.1, line_width=3, line_dash="dash", line_color="orange", annotation_text="Moyenne")
    fig.update_layout(title=f'Taux de difficulté de paiement (%) par {var}')
    
    return fig, value


def plot_distribution(df, var, sk_id, n_bins=10):
    
    x1 = df.loc[df['TARGET'] == 0, var].values
    x1 = x1[~np.isnan(x1)]
    x2 = df.loc[df['TARGET'] == 1, var].values
    x2 = x2[~np.isnan(x2)]
    
    #group_labels = ['Pas de difficulté de paiement', 'Difficulté de paiement']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x1, nbinsx = n_bins, name='Sans difficulté'))
    fig.add_trace(go.Histogram(x=x2, nbinsx = n_bins, name='Difficultés'))
    
    value = df.loc[sk_id][var]
    fig.add_vline(x=value, line_width=5,
                  line_dash="dash", line_color="purple",
                  annotation_text=f"client du prêt {sk_id}")
    
    fig.update_layout(barmode='stack')
    fig.update_layout(title=f'Distribution de {var}')
    
    return fig


def plot_repartition(df, var, sk_id):
    
    total_tab = df[var].value_counts()
    labels = total_tab.index
    values = total_tab.values
    
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title=f'Répartiton des prêts par {var}')
    
    return fig