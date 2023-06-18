from django.shortcuts import render
from .forms import StockForm
import yfinance as yf
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

df = pd.read_csv('dashboard/models/df.csv')
X = pd.read_csv('dashboard/models/X.csv')

with open('dashboard/models/rfr.pkl', 'rb') as f:
    rfr_model = pickle.load(f)
    
with open('dashboard/models/svr.pkl', 'rb') as f:
    svr_model = pickle.load(f)

with open('dashboard/models/voting.pkl', 'rb') as f:
    voting_model = pickle.load(f)
    
with open('dashboard/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def dashboard(request):
    SelectedStock = None
    form = StockForm()

    if request.method == "POST":
        form = StockForm(request.POST)

        if form.is_valid():
            userType = form.cleaned_data['userType']

            if userType == 'basic':
                SelectedStock = form.cleaned_data['name']

                last_index_date = df['Date'].tail(1).astype('str').values[0]
                start_index_date = df['Date'].tail(30).head(1).astype('str').values[0]

                # Download latest 30 days historical price data from Yahoo Finance and store in a pandas DataFrame
                df_actual = yf.download("GOOGL", start=start_index_date, end=last_index_date, progress=False)
            
                # number_of_days = len(df_actual)
                # last_n_days = df[-number_of_days:]

                # last_n_days_df = pd.DataFrame(
                #     last_n_days, columns=X.columns)

                # X_pred = scaler.transform(last_n_days_df)

                # y_svr_pred = svr_model.predict(X_pred) 
                # y_rfr_pred = rfr_model.predict(X_pred)
                # y_voting_pred = voting_model.predict(X_pred)

                # df_pred = pd.DataFrame({'SVR Prediction': y_svr_pred,
                #         'RFR Prediction': y_rfr_pred,
                #         'Voting Prediction': y_voting_pred}, index=df_actual.index)

                # df_combined = pd.concat([df_actual, df_pred], axis=1)

                # forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(days=1), periods=10, freq='B')
                # df_forecast = pd.DataFrame(index=forecast_dates)

                # # Get the last available historical data
                # last_data = X.iloc[-10:, :]  # Get the last 35 data points, assuming each row represents a sample

                # # Preprocess the forecast data by scaling it using the same scaler used for training
                # last_data_scaled = scaler.transform(last_data)

                # # Make predictions for the forecast period using the SVR and RFR models
                # svr_prediction = svr_model.predict(last_data_scaled)
                # rfr_prediction = rfr_model.predict(last_data_scaled)
                # voting_prediction = voting_model.predict(last_data_scaled)

                # # Assign the forecasted prices to the DataFrame columns
                # df_forecast['SVR Prediction'] = svr_prediction
                # df_forecast['RFR Prediction'] = rfr_prediction
                # df_forecast['Voting Prediction'] = voting_prediction

                # # Set the display format for float values
                # pd.set_option('display.float_format', lambda x: '%.3f' % x)

                # df_combined2 = pd.concat([df_combined, df_forecast])

                # # region Forecast
                # forecast_graphs = []
                # forecast_graphs.append(
                #     go.Scatter(
                #         x=df_combined2.index,
                #         y=df_combined2['Close'],
                #         mode='lines',
                #         name='Historical'
                #     )
                # )

                # forecast_graphs.append(
                #     go.Scatter(
                #         x=df_combined2.index,
                #         y=df_combined2['SVR Prediction'],
                #         mode='lines',
                #         name='SVR Prediction'
                #     )             
                # )

                # forecast_graphs.append(
                #     go.Scatter(
                #         x=df_combined2.index,
                #         y=df_combined2['RFR Prediction'],
                #         mode='lines',
                #         name='RFR Prediction'
                #     )             
                # )

                # forecast_graphs.append(
                #     go.Scatter(
                #         x=df_combined2.index,
                #         y=df_combined2['Voting Prediction'],
                #         mode='lines',
                #         name='Voting Prediction'
                #     )             
                # )

                # # Create layout
                # layout = go.Layout(
                #     title='Actual Price and Predictions',
                #     xaxis=dict(title='Date'),
                #     yaxis=dict(title='Price'),
                #     showlegend=True
                # )

                # # svr_layout = {
                # #     'title': 'Support Vector Regression (SVR)',
                # #     'xaxis_title': 'Year',
                # #     'yaxis_title': 'Price',
                # #     'height': 300,
                # #     'width': 500,
                # # }

                # # Getting HTML needed to render the plot.
                # forecast = plot({'data': forecast_graphs, 'layout': layout},
                #             output_type='div')

                # # endregion

                context = {'form': form,
                'SelectedStock': SelectedStock,
                'userType': userType,
                # 'forecast': forecast,
                # 'sk_ridge': sk_ridge,
                # 'sk_knr': sk_knr,
                # 'sk_rfr': sk_rfr, 
                }
            



            else:
                SelectedStock = form.cleaned_data['name']
                print(SelectedStock, userType)

                context = {'form': form,
                'SelectedStock': SelectedStock,
                'userType': userType,
                # 'sk_svr': sk_svr,
                # 'sk_ridge': sk_ridge,
                # 'sk_knr': sk_knr,
                # 'sk_rfr': sk_rfr, 
                }

        return render(request, 'dashboard/dashboard.html', context)

    context = {'form': form, 'SelectedStock': SelectedStock}
    return render(request, 'dashboard/dashboard.html', context)

