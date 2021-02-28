import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv(r'C:\Users\Neelam\Desktop\Codes\forecast\train.csv',  
                    low_memory = False)

df = df[(df["Open"] != 0) & (df['Sales'] != 0)]

# sales for the store number 1 (StoreType C)
sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending = False)

sales.head()
sales.Date.min() 
sales.Date.max()

sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales.dtypes

sales = sales.rename(columns = {'Date': 'ds',
                                'Sales': 'y'})
sales.head()



#my_model = Prophet(interval_width = 0.95
#                   )
#my_model.fit(sales)

#future = my_model.make_future_dataframe(periods=6*7)
#forecast = my_model.predict(future)

#plot = my_model.plot(forecast)

from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
#cross_validation_results = cross_validation(my_model, initial='210 days', period='15 days', horizon='70 days')
#print(cross_validation_results)

#performance_metrics_results = performance_metrics(cross_validation_results)
#print(performance_metrics_results)

#################
#taking into consideration , holidays

# create holidays dataframe
state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b') & (df.StateHoliday == 'c')].loc[:, 'Date'].values
school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                      'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))      
holidays.head()

model = Prophet(interval_width = 0.95,weekly_seasonality=True,changepoint_prior_scale=0.5,
                   holidays = holidays)
#model.add_seasonality(name='monthly', period=30.5,fourier_order=5)
model.fit(sales)
#changepoint_prior_scale=0.5
#weekly_seasonality=True
#m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# dataframe that extends into future 6 weeks 
future_dates = model.make_future_dataframe(periods = 6*7)

print("First week to forecast.")
future_dates.tail(7)

# predictions
forecast = model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})

model.plot(forecast);
'''
plt.plot(sales['y'], marker='.', label="true")
plt.plot(fc['yhat'], 'r', label="prediction")
plt.ylabel('Value')
plt.xlabel('Time Step')
plt.legend()
plt.show();
'''
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
cross_validation_results = cross_validation(model, initial='210 days', period='15 days', horizon='70 days')
print(cross_validation_results)

performance_metrics_results2 = performance_metrics(cross_validation_results)
print(performance_metrics_results2)

from fbprophet.plot import plot_cross_validation_metric
fig3 = plot_cross_validation_metric(cross_validation_results, metric='mape')

from fbprophet.plot import plot_cross_validation_metric
fig3 = plot_cross_validation_metric(df_cv, metric='mape')
# Python
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(model, forecast)  # This returns a plotly Figure
py.iplot(fig)