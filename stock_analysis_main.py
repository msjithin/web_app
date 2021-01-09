
from os import write
from PIL import Image
import streamlit as st 
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from streamlit.proto import ColorPicker_pb2
from stock_analysis import *
from streamlit.elements import button
# Use the full page instead of a narrow central column
#st.set_page_config(page_title='Stock web app')

image1, image2, image3 = st.beta_columns([2,1,1])
#Add title and image 
image1.write("""
# Stock Market Analysis
""")
top_image = Image.open('assets/stockimage.jpg')
image2.image(top_image, use_column_width=True)

#Create site bar header
st.sidebar.header('User Input')
#add select widget
st.sidebar.subheader('Select Stock')
stock_names = ['Microsoft', 'Apple', 'S&P 500']
dropdown_select_stock = st.sidebar.selectbox(label='Stock name', options=stock_names)
stock_selected = 'MSFT'
if dropdown_select_stock == 'Microsoft':
    stock_selected = 'MSFT'
elif dropdown_select_stock == 'Apple':
    stock_selected = 'AAPL'
elif dropdown_select_stock == 's&P 500':
    stock_selected = '^GSPC'
st.write(get_time_series(stock_selected))
st.write(get_line_plotly(stock_selected))
st.subheader('Log returns')
st.write(plot_return())

def weights_pie_chart(donut=True, weights=weights_vector.transpose().values[-1].tolist()):
    labels = stock_names
    sizes = weights
    # print('weights = ', sizes)
    # # Use `hole` to create a donut-like pie chart
    layout = go.Layout(height = 400, width = 400,
        autosize = True
        )
    if donut:
        fig1 = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3)], layout = layout)
    else:
        fig1 = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent',
                             insidetextorientation='radial')] ,layout = layout)
    
    return fig1

st.subheader('Weights')    
pie1, pie2 = st.beta_columns([1,1])
input_weight = [1]*len(stock_names)
for i in range(len(stock_names)):
    input_weight[i] = pie1.slider( stock_names[i] + ' weight', value=1.0, min_value=0.0, max_value=1.0, step=0.1, key=i+1)
if sum(input_weight) == 0:
    normalized_weight = [1, 1, 1]
else:
    normalized_weight = [ x / sum(input_weight) for x in input_weight ]
pie2.write(weights_pie_chart(False, normalized_weight))

st.subheader('Portfolio returns')
st.write(portfolio_returns())

st.sidebar.write('**Total portfolio return is** : ' )
st.sidebar.success('{:5.2f}'.format(100 * total_portfolio_return) + '%')
st.sidebar.write('**Average yearly return is**  : ' )
st.sidebar.success('{:5.2f}'.format(100 * average_yearly_return) + '%')



st.subheader('Select Date range ')
date_c1, date_c2 = st.beta_columns([1, 1])
din = date_c1.date_input( "Start date", dt.date(2015, 7, 6))
dout = date_c2.date_input( "End date", dt.date(2019, 7, 6))

st.subheader(stock_selected + '  Price')
st.write(plot_price(stock_selected, din, dout))

st.subheader(stock_selected + '  EMA')
st.write(plot_ema(stock_selected, din, dout))

st.subheader(stock_selected + '  Trading position')
st.write(plot_trading_position(stock_selected, din, dout))

st.subheader('Best return ')
st.write(plot_best_returns())

st.subheader('Total returns ')
st.write(total_retuns())

#print(dt.datetime.now().strftime("%H:%M:%S") , ' script end reached .....')