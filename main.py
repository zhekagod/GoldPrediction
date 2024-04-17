import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta, date
import streamlit.components.v1 as components
import time as time

html_string = '''
<marquee>Gold price prediction</marquee>
'''
JavaScriptIframe = '''
<style>
  html,body {
    height: 100%;
}
body {
    //background: #0f3854;
    //background: radial-gradient(ellipse at center,  #0a2e38  0%, #000000 70%);
    background-size: 100%;
}
p {
    margin: 0;
    padding: 0;
}
#clock {
    font-family: 'Share Tech Mono', monospace;
    color: #ffffff;
    text-align: center;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: #daf6ff;
    text-shadow: 0 0 20px rgba(10, 175, 230, 1),  0 0 20px rgba(10, 175, 230, 0);
    .time {
        letter-spacing: 0.05em;
        font-size: 60px;
        padding: 5px 0;
    }
    .date {
        letter-spacing: 0.1em;
        font-size: 24px;
    }
}
</style>
<div id="clock">
    <p class="date" id="date"></p>
    <p class="time" id="time"></p>
</div>

<script>
function updateTime() {
    var cd = new Date();
    var time = document.getElementById("time");
    var date = document.getElementById("date");

    var week = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];

    var hours = zeroPadding(cd.getHours(), 2);
    var minutes = zeroPadding(cd.getMinutes(), 2);
    var seconds = zeroPadding(cd.getSeconds(), 2);

    var year = zeroPadding(cd.getFullYear(), 4);
    var month = zeroPadding(cd.getMonth() + 1, 2);
    var day = zeroPadding(cd.getDate(), 2);
    var dayOfWeek = week[cd.getDay()];

    time.textContent = hours + ':' + minutes + ':' + seconds;
    date.textContent = year + '-' + month + '-' + day + ' ' + dayOfWeek;
}

function zeroPadding(num, digit) {
    var zero = '';
    for (var i = 0; i < digit; i++) {
        zero += '0';
    }
    return (zero + num).slice(-digit);
}

setInterval(updateTime, 1000);
updateTime();
  </script>'''
components.html(JavaScriptIframe)  # JavaScript works

st.markdown(html_string, unsafe_allow_html=True) 

#st.sidebar.title('Дата')#боковая панель
#st.cache # для оптимизации работы приложения

# Определение текущей даты
# current_date = datetime.now().strftime("%d-%m-%y")
# st.sidebar.markdown(f'Текущая дата: {current_date}')

st.markdown(
    """
    <style>
    .stDateInput {
      display: flex;
      flex-direction: column;
      max-width: 120px;
    }
    #ded33bd3, #c7762039 {
      font-size: 1.2rem;
    }

    .st-emotion-cache-eqffof.e1nzilvr5 p {
      font-size: 1.4rem;
      margin-bottom: 0;
    }

   
     .st-emotion-cache-vk3wp9.eczjsme11 {
      background: rgba(217, 217, 217, 0.2);
      backdrop-filter: blur(40px);
      -webkit-backdrop-filter: blur(40px);
      box-shadow: rgba(0, 0, 0, 0.2) 0 0 20px;
      box-shadow: rgba(0, 0, 0, 0.2) 0 0 20px;
    }
    
    .st-emotion-cache-10trblm, p, h1 {
        color: white;
    }
    .stApp {
    background: rgb(2,0,36);
background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 52%, rgba(50,169,193,1) 100%);
 }
 .st-emotion-cache-13ejsyy.ef3psqc12 {
  background-color: #1E1F21;
 }

   .st-emotion-cache-1kyxreq.e115fcil2 {
    display: flex;
    justify-content: center;
   }
    </style>
    """,
    unsafe_allow_html=True
  )

st.sidebar.image("images/CoinPngSmall.gif")
st.sidebar.audio("audio/archivo.mp3")

# if st.sidebar.button('Темная тема'):
#   st.markdown(
#     """
#     <style>
#     //.st-emotion-cache-1r4qj8v {
#     //    background-color: #31333F;
#    // }
#     .st-emotion-cache-10trblm, p, h1 {
#         color: white;
#     }
#     .stApp {
#     background: rgb(2,0,36);
# background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 52%, rgba(50,169,193,1) 100%);
#  }
#  .st-emotion-cache-13ejsyy.ef3psqc12 {
#   background-color: #1E1F21;
#  }
#     </style>
#     """,
#     unsafe_allow_html=True
#   )

# if st.sidebar.button('Светлая тема'):
#   st.markdown(
#     """
#     <style>
#     iframe #clock {
#     color: black;
#     }
#   //  .st-emotion-cache-1r4qj8v {
#   //      background-color: white;
#   //  }
#     .st-emotion-cache-10trblm, p, h1 {
#         color: #31333F;
#     }

#     .stApp {
#     background: rgb(242,73,73);
# background: linear-gradient(90deg, rgba(242,73,73,1) 0%, rgba(216,212,72,1) 49%, rgba(244,176,81,1) 100%);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
#   )
#31333F
# Чтение данных из файла
data = pd.read_csv('datas\GC_230314_240314.csv', delimiter=';')

# Преобразование столбца с датой в формат datetime
data['<DATE_TIME>'] = pd.to_datetime(data['<DATE>'].astype(str) + ' ' + data['<TIME>'].astype(str), format='%y%m%d %H%M%S')
#data['<DATE>'] = pd.to_datetime(data['<DATE>'] + ' ' + data['<TIME>'])
data.set_index('<DATE_TIME>', inplace=True)
# Создание веб-приложения с использованием Streamlit
st.title('Графики цен')

choosen_date = st.date_input("Выберите дату:", value = datetime.now().date(), 
  min_value = date(2011, 1, 1), 
  max_value = (datetime.now() + timedelta(days=30)).date())

df = pd.DataFrame(data)

filtered_df = data[data.index.date == choosen_date]

if not filtered_df.empty:
    with st.spinner(text='In progress'):
      time.sleep(2)
    st.success('Done')
    st.subheader("Реальная цена на выбранную дату:")
    st.write(filtered_df['<OPEN>'].iloc[0])  # Первое значение графика в выбранную дату
    st.divider()
    st.subheader("Прогнозируемая цена на выбранную дату:")
    st.write(filtered_df['<CLOSE>'].iloc[0])  # Второе значение графика в выбранную дату
    st.divider()
else:
    st.write('Данные для выбранной даты отсутствуют')

chart_type = st.selectbox('Выберите тип графика:', ['Цена открытия', 'Цена закрытия', 'Объем торгов'])

if chart_type == 'Цена открытия':
  st.subheader('График цены открытия')
  st.line_chart(data['<OPEN>'])
elif chart_type == 'Цена закрытия':
  st.subheader('График цены закрытия')
  st.line_chart(data['<CLOSE>'])
else:
  st.subheader('График объема торгов')
  st.line_chart(data['<VOL>'])

# chat_date_str = st.chat_input("Write the date")
# chat_date = datetime.now().date()
# if(chat_date_str):
#   chat_date = datetime.strptime(chat_date_str, '%Y-%m-%d').date()

# chat_df = data[data.index.date == chat_date]

# with st.chat_message("user"):
#     st.write(f"Estimated cost on {chat_date}:")
#     if not chat_df.empty:
#         st.write(chat_df['<CLOSE>'].iloc[0])
#     else:
#       st.write('Данные для выбранной даты отсутствуют')
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['<OPEN>'],
                high=data['<HIGH>'],
                low=data['<LOW>'],
                close=data['<CLOSE>'])])

# Настройка цветов свечей в зависимости от изменения цены
fig.update_layout(xaxis_rangeslider_visible=False)  # Убираем ползунок для увеличения масштаба на оси x
fig.update_traces(selector=dict(type='candlestick'), 
                  increasing_line_color='green', 
                  decreasing_line_color='red')

buttons_to_add = [
    'drawline',
    'drawopenpath',
    'drawclosedpath',
    'drawcircle',
    'drawrect',
    'eraseshape'
]

fig.update_layout(
    title='Свечной график цен на золото',
    xaxis_title='Дата',
    yaxis_title='Цена',
    dragmode='zoom',
    modebar_add=buttons_to_add
)

st.plotly_chart(fig)  # Перерисовываем график с обновленными настройками