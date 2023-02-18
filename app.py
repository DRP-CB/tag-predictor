from dash import Dash, dcc, html, Input, Output, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import plotly.express as px

tags =  pd.read_csv('tags_df.csv',
                    index_col=0)
USE_model = tf.saved_model.load('universal-sentence-encoder_4/')
predictor = tf.keras.models.load_model('USE_model.h5')


app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server
app.layout = dbc.Container(
    [
        dbc.Row(html.H1('Prédiction de tags')),
        dbc.Row(html.H6('Entrer un titre et une question comme sur stack overflow puis cliquer sur le bouton "prédire" pour estimer des tags')),
        html.Br(),
        html.H3('Titre'),
        dbc.Row(dcc.Input(id="titre")),
        html.H3('Question'),
        dbc.Row(dcc.Input(id='question')),
        html.Br(),
        dbc.Row(dbc.Button('Prédire',
                            id='get_predict',
                            className="me-1")),
        dcc.Store(id='store'),
        html.Div(id='test_output')
    ])


@app.callback(Output('store','data'),
              [Input('titre','value'),
               Input('question','value')])
def return_content(titre,question):
    
        if titre is not None and titre != '' and question is not None and question != '':
            return str(titre + ' ' + question)
        else :
            return ""


@app.callback(Output('test_output','children'),
              [Input('store','data'),
               Input('get_predict','n_clicks')])
def predict(input1,n_clicks):
    if 'get_predict' == ctx.triggered_id:
        if input1 is None or input1 == "":
            return "Les deux champs doivent être remplis pour la prédiction."
        elif input1 is not None or input1 != "":
            prediction = predictor(USE_model([input1])).numpy().ravel()
            tags_prediction = tags.copy()
            tags_prediction['prediction'] = prediction
            tags_prediction = tags_prediction.sort_values('prediction',ascending=False).head(5)
            fig = px.bar(tags_prediction,
                          x='prediction',
                          y='tags',
                          orientation='h',
                          color='tags')
            

            return dcc.Graph(id="figure",figure=fig)
        
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
