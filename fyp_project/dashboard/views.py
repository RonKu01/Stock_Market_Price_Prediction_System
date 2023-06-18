from django.shortcuts import render
from .forms import StockForm
import yfinance as yf
import pickle
import pandas as pd

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
                print(SelectedStock, userType)

                df = yf.download(
                    SelectedStock, start="2019-09-01", end="2022-09-01")
                df = df.asfreq("B", method="pad").sort_index()

                test = round(len(df) * 0.1)
                data_test = df.iloc[-test:, :]

                real_data = yf.download(
                    SelectedStock, start="2022-09-01", end="2022-12-01")
                real_data = real_data.iloc[:, :].asfreq("B").sort_index()
                real_data = real_data.bfill().ffill()
                df2 = real_data.drop(
                    ['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

                # region SVR configuration
                sk_svr = load('dashboard/ML_Models/ml_models_3years/sk_svr.pkl')
                last_window = data_test['Close'][-sk_svr.window_size:]
                prediction = sk_svr.predict(steps=len(real_data), exog=real_data[[
                                            'Open', 'Adj Close', 'High', 'Low', 'Volume']], last_window=last_window)
                df2['Prediction'] = prediction

                svr_graphs = []
                svr_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Prediction'], mode='lines', name='Prediction')
                )
                svr_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Close'], mode='lines', name='Close')
                )
                svr_layout = {
                    'title': 'Support Vector Regression (SVR)',
                    'xaxis_title': 'Year',
                    'yaxis_title': 'Price',
                    'height': 300,
                    'width': 500,
                }

                # Getting HTML needed to render the plot.
                sk_svr = plot({'data': svr_graphs, 'layout': svr_layout},
                            output_type='div')

                # endregion

                # region Ridge configuration
                sk_ridge = load(
                    'dashboard/ML_Models/ml_models_3years/sk_ridge.pkl')
                last_window = data_test['Close'][-sk_ridge.window_size:]
                prediction = sk_ridge.predict(steps=len(real_data), exog=real_data[[
                    'Open', 'Adj Close', 'High', 'Low', 'Volume']], last_window=last_window)
                df2['Prediction'] = prediction

                ridge_graphs = []
                ridge_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Prediction'], mode='lines', name='Prediction')
                )
                ridge_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Close'], mode='lines', name='Close')
                )
                ridge_layout = {
                    'title': 'Ridge Regression',
                    'xaxis_title': 'Year',
                    'yaxis_title': 'Price',
                    'height': 300,
                    'width': 500,
                }

                # Getting HTML needed to render the plot.
                sk_ridge = plot({'data': ridge_graphs, 'layout': ridge_layout},
                                output_type='div')

                # endregion

                # region KNR configuration
                sk_knr = load(
                    'dashboard/ML_Models/ml_models_3years/sk_knr.pkl')
                last_window = data_test['Close'][-sk_knr.window_size:]
                prediction = sk_knr.predict(steps=len(real_data), exog=real_data[[
                    'Open', 'Adj Close', 'High', 'Low', 'Volume']], last_window=last_window)
                df2['Prediction'] = prediction

                knr_graphs = []
                knr_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Prediction'], mode='lines', name='Prediction')
                )
                knr_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Close'], mode='lines', name='Close')
                )
                knr_layout = {
                    'title': 'K-Nearest Neighbors Regression (KNR)',
                    'xaxis_title': 'Year',
                    'yaxis_title': 'Price',
                    'height': 300,
                    'width': 500,
                }

                # Getting HTML needed to render the plot.
                sk_knr = plot({'data': knr_graphs, 'layout': knr_layout},
                            output_type='div')

                # endregion

                # region RFR configuration
                sk_rfr = load(
                    'dashboard/ML_Models/ml_models_3years/sk_rfr.pkl')
                last_window = data_test['Close'][-sk_rfr.window_size:]
                prediction = sk_rfr.predict(steps=len(real_data), exog=real_data[[
                    'Open', 'Adj Close', 'High', 'Low', 'Volume']], last_window=last_window)
                df2['Prediction'] = prediction

                rfr_graphs = []
                rfr_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Prediction'], mode='lines', name='Prediction')
                )
                rfr_graphs.append(
                    go.Scatter(
                        x=df2.index, y=df2['Close'], mode='lines', name='Close')
                )
                rfr_layout = {
                    'title': 'Random Forest Regression (RFR)',
                    'xaxis_title': 'Year',
                    'yaxis_title': 'Price',
                    'height': 300,
                    'width': 500,
                }

                # Getting HTML needed to render the plot.
                sk_rfr = plot({'data': rfr_graphs, 'layout': rfr_layout},
                            output_type='div')

                # endregion




                

                context = {'form': form,
                'SelectedStock': SelectedStock,
                'userType': userType,
                # 'sk_svr': sk_svr,
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

