from django.shortcuts import render
from .forms import StockForm
import yfinance as yf
import pickle
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go
from plotly.offline import plot


def dashboard(request):
    form = StockForm()
    SelectedStock = 'GOOGL'
    ml_model = 'voting_model'
    num_of_days = 10

    df = pd.read_csv('dashboard/models/google/df.csv')
    X = pd.read_csv('dashboard/models/google/X.csv')

    with open('dashboard/models/google/rfr.pkl', 'rb') as f:
        rfr_model = pickle.load(f)
        
    with open('dashboard/models/google/svr.pkl', 'rb') as f:
        svr_model = pickle.load(f)

    with open('dashboard/models/google/voting.pkl', 'rb') as f:
        voting_model = pickle.load(f)
        
    with open('dashboard/models/google/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    if request.method == "POST":
        form = StockForm(request.POST)
        
        if form.is_valid(): # Expert User
            ml_model = form.cleaned_data['ml_model']
            num_of_days = form.cleaned_data['num_of_days']

        SelectedStock = form.cleaned_data['name']

        if SelectedStock == 'GOOGL':
            df = pd.read_csv('dashboard/models/google/df.csv')
            X = pd.read_csv('dashboard/models/google/X.csv')

            with open('dashboard/models/google/rfr.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
                
            with open('dashboard/models/google/svr.pkl', 'rb') as f:
                svr_model = pickle.load(f)

            with open('dashboard/models/google/voting.pkl', 'rb') as f:
                voting_model = pickle.load(f)
                
            with open('dashboard/models/google/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        elif SelectedStock == 'AAPL':
            df = pd.read_csv('dashboard/models/apple/df.csv')
            X = pd.read_csv('dashboard/models/apple/X.csv')

            with open('dashboard/models/apple/rfr.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
                
            with open('dashboard/models/apple/svr.pkl', 'rb') as f:
                svr_model = pickle.load(f)

            with open('dashboard/models/apple/voting.pkl', 'rb') as f:
                voting_model = pickle.load(f)
                
            with open('dashboard/models/apple/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        else:
            df = pd.read_csv('dashboard/models/microsoft/df.csv')
            X = pd.read_csv('dashboard/models/microsoft/X.csv')

            with open('dashboard/models/microsoft/rfr.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
                
            with open('dashboard/models/microsoft/svr.pkl', 'rb') as f:
                svr_model = pickle.load(f)

            with open('dashboard/models/microsoft/voting.pkl', 'rb') as f:
                voting_model = pickle.load(f)
                
            with open('dashboard/models/microsoft/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

    last_index_date = df['Date'].tail(1).astype('str').values[0]
    start_index_date = df['Date'].tail(30).head(1).astype('str').values[0]

    # Download latest 30 days historical price data from Yahoo Finance and store in a pandas DataFrame
    df_actual = yf.download(SelectedStock, start=start_index_date, end=last_index_date, progress=False)

    number_of_days = len(df_actual)
    last_n_days = df[-number_of_days:]

    last_n_days_df = pd.DataFrame(
        last_n_days, columns=X.columns)

    last_n_days_df = last_n_days_df.set_index('Date')

    X_pred = scaler.transform(last_n_days_df)

    y_svr_pred = svr_model.predict(X_pred) 
    y_rfr_pred = rfr_model.predict(X_pred)
    y_voting_pred = voting_model.predict(X_pred)

    df_pred = pd.DataFrame({'SVR Prediction': y_svr_pred,
            'RFR Prediction': y_rfr_pred,
            'Voting Prediction': y_voting_pred}, index=df_actual.index)

    df_combined = pd.concat([df_actual, df_pred], axis=1)

    # Convert last_index_date to a pandas Timestamp object
    last_index_date = pd.Timestamp(last_index_date)

    # Add a DateOffset of 1 day to last_index_date and convert it to a string
    next_day = (last_index_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

    # Generate the forecast dates using the updated next_day
    forecast_dates = pd.date_range(start=next_day, periods=num_of_days, freq='B')
    df_forecast = pd.DataFrame(index=forecast_dates)

    # Get the last available historical data
    last_data = X.iloc[-num_of_days:, :]  # Get the last 35 data points, assuming each row represents a sample
    
    last_data = last_data.set_index('Date')
    # Preprocess the forecast data by scaling it using the same scaler used for training
    last_data_scaled = scaler.transform(last_data)

    # Make predictions for the forecast period using the SVR and RFR models
    svr_prediction = svr_model.predict(last_data_scaled)
    rfr_prediction = rfr_model.predict(last_data_scaled)
    voting_prediction = voting_model.predict(last_data_scaled)

    # Assign the forecasted prices to the DataFrame columns
    df_forecast['SVR Prediction'] = svr_prediction
    df_forecast['RFR Prediction'] = rfr_prediction
    df_forecast['Voting Prediction'] = voting_prediction

    # Set the display format for float values
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    df_combined2 = pd.concat([df_combined, df_forecast])

    # region Forecast
    forecast_graphs = []

    forecast_graphs.append(
        go.Scatter(
            x=df_combined2.index,
            y=df_combined2['Close'],
            mode='lines',
            name='Historical'
        )
    )

    if ml_model == 'svr_model':
        forecast_graphs.append(
            go.Scatter(
                x=df_combined2.index,
                y=df_combined2['SVR Prediction'],
                mode='lines',
                name='SVR Prediction'
            )             
        )



        nextpredict = df_forecast['SVR Prediction'][0]
        fivedayspredict = df_forecast['SVR Prediction'][4]
        tendayspredict = df_forecast['SVR Prediction'][9]

    if ml_model == 'rfr_model':
        forecast_graphs.append(
            go.Scatter(
                x=df_combined2.index,
                y=df_combined2['RFR Prediction'],
                mode='lines',
                name='RFR Prediction'
            )             
        )

        nextpredict = df_forecast['RFR Prediction'][0]
        fivedayspredict = df_forecast['RFR Prediction'][4]
        tendayspredict = df_forecast['RFR Prediction'][9]

    if ml_model == 'voting_model':
        forecast_graphs.append(
            go.Scatter(
                x=df_combined2.index,
                y=df_combined2['Voting Prediction'],
                mode='lines',
                name='Voting Prediction'
            )             
        )

        nextpredict = df_forecast['Voting Prediction'][0]
        fivedayspredict = df_forecast['Voting Prediction'][4]
        tendayspredict = df_forecast['Voting Prediction'][9]

    # Create layout
    forecast_graphs_layout = go.Layout(
        title='Actual Price and Predictions',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        showlegend=True,
        title_font=dict(size=24, family='Arial', color='black')
    )

    # Getting HTML needed to render the plot.
    forecast = plot({'data': forecast_graphs, 'layout': forecast_graphs_layout},
                output_type='div')
    
    # endregion

    # region Volume
    volume_history = df_combined2['Volume']
    weekly_volume = volume_history.resample('W').sum()


    volume_graph = go.Bar(
                    x=weekly_volume.index,
                    y=weekly_volume.values,
                    name='Volume'
                )
    
    # Create layout
    volume_graphs_layout = go.Layout(
        title='Volume History for ' + SelectedStock,
        xaxis=dict(title='Date'),
        yaxis=dict(title='Volume'),
        showlegend=True,
        title_font=dict(size=25, family='Arial', color='black'),
    )

    volume = plot({'data': volume_graph, 'layout': volume_graphs_layout}, output_type='div')

    # endregion

    # region Recommendation
    selectedstock = Ticker(SelectedStock)
    recommendation_trend = selectedstock.recommendation_trend

    strong_buy = recommendation_trend['strongBuy'][0]
    buy = recommendation_trend['buy'][0]
    hold = recommendation_trend['hold'][0]
    sell = recommendation_trend['sell'][0]
    strong_sell = recommendation_trend['strongSell'][0]

    total_recommendations = strong_buy + buy + hold + sell + strong_sell
    value = (buy + strong_buy) / total_recommendations

    indicator = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': "Recommendation Trend"},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': 'rgba(50, 175, 255, 0.7)'},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.2], 'color': 'rgba(255, 0, 0, 0.6)', 'name': 'Strong Sell'},
                    {'range': [0.2, 0.4], 'color': 'rgba(255, 100, 0, 0.6)', 'name': 'Sell'},
                    {'range': [0.4, 0.6], 'color': 'rgba(255, 255, 0, 0.6)', 'name': 'Neutral'},
                    {'range': [0.6, 0.8], 'color': 'rgba(0, 255, 0, 0.6)', 'name': 'Buy'},
                    {'range': [0.8, 1], 'color': 'rgba(0, 200, 0, 0.6)', 'name': 'Strong Buy'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 5},
                    'thickness': 0.75,
                    'value': value
                }
            }
        )
    )

    annotations = []

    if value < 0.2:
        annotations.append(
            dict(x=0.5, y=0.5, text="Strong Sell", showarrow=False, font={'size': 12, 'color': 'red'})
        )
    elif value < 0.4:
        annotations.append(
            dict(x=0.5, y=0.5, text="Sell", showarrow=False, font={'size': 12, 'color': 'orange'})
        )
    elif value < 0.6:
        annotations.append(
            dict(x=0.5, y=0.5, text="Neutral", showarrow=False, font={'size': 12, 'color': 'gold'})
        )
    elif value < 0.8:
        annotations.append(
            dict(x=0.5, y=0.5, text="Buy", showarrow=False, font={'size': 12, 'color': 'lightgreen'})
        )
    else:
        annotations.append(
            dict(x=0.5, y=0.5, text="Strong Buy", showarrow=False, font={'size': 12, 'color': 'green'})
        )

    layout = go.Layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=annotations
    )

    indicator.update_layout(layout)

    # Getting HTML needed to render the plot.
    recommendation = plot(indicator, output_type='div')

    # endregion

    last_close_price = df_actual['Close'][-1]

    nextpredict = round(nextpredict, 2)
    fivedayspredict = round(fivedayspredict, 2)
    tendayspredict = round(tendayspredict, 2)

    nextpredict_change = round(((nextpredict - last_close_price) / last_close_price) * 100, 2)
    fivedayspredict_change = round(((fivedayspredict - last_close_price) / last_close_price) * 100, 2)
    tendayspredict_change = round(((tendayspredict - last_close_price) / last_close_price) * 100, 2)
    
    context = { 'form'            : form,
                'SelectedStock'   : SelectedStock,
                'forecast'        : forecast,
                'nextpredict'     : nextpredict,
                'fivedayspredict' : fivedayspredict,
                'tendayspredict'  : tendayspredict,
                'nextpredict_change': nextpredict_change,
                'fivedayspredict_change':fivedayspredict_change,
                'tendayspredict_change':tendayspredict_change,
                'volume'          : volume,
                'recommendation'  : recommendation,
            }
            
    return render(request, 'dashboard/dashboard.html', context)

