import pandas as pd
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


# Define start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Prediction App")

# Corrected stock tuple (flattened)
stocks = [
    "AAPL", "GOOG", "MSFT", "GME", "RELIANCE.NS", "INFY.BO", "JIOFIN.BO",
    "TCS.BO", "HDFCBANK.BO", "ICICIBANK.BO", "HINDUNILVR.BO",
    "SBIN.BO", "BHARTIARTL.BO", "KOTAKBANK.BO", "BAJFINANCE.BO","BAJFINANCE.NS", "ITC.BO", "LICI.BO",
    "ADANIGREEN.BO", "ADANITRANS.BO", "ADANIPORTS.BO", "ADANIENT.BO", "LT.BO",
    "HCLTECH.BO", "AXISBANK.BO", "ASIANPAINT.BO", "DMART.BO", "MARUTI.BO", "SUNPHARMA.BO",
    "TITAN.BO", "ONGC.BO", "ULTRACEMCO.BO", "BAJAJFINSV.BO", "WIPRO.BO", "POWERGRID.BO",
    "HINDZINC.BO", "JSWSTEEL.BO", "NTPC.BO", "COALINDIA.BO", "TATAMOTORS.BO", "M&M.BO",
    "VEDL.BO", "SBILIFE.BO", "DIVISLAB.BO", "TECHM.BO", "DABUR.BO", "HDFCLIFE.BO",
    "NESTLEIND.BO", "PIDILITIND.BO", "GRASIM.BO", "TATASTEEL.BO", "HEROMOTOCO.BO",
    "DRREDDY.BO", "BPCL.BO", "EICHERMOT.BO", "BRITANNIA.BO", "BAJAJ-AUTO.BO", "SHREECEM.BO",
    "GODREJCP.BO", "ICICIPRULI.BO", "MOTHERSUMI.BO", "SIEMENS.BO", "CIPLA.BO", "GAIL.BO",
    "SRF.BO", "INDUSINDBK.BO", "IOC.BO", "HAVELLS.BO", "LTI.BO", "PGHH.BO", "COLPAL.BO",
    "TORNTPHARM.BO", "MCDOWELL-N.BO", "BIOCON.BO", "BANDHANBNK.BO", "AMBUJACEM.BO",
    "BERGEPAINT.BO", "PIIND.BO", "NAUKRI.BO", "INDIGO.BO", "ACC.BO", "ADANIPOWER.BO",
    "GLAND.BO", "TATACONSUM.BO", "JUBLFOOD.BO", "APOLLOHOSP.BO", "BOSCHLTD.BO",
    "HINDPETRO.BO", "NMDC.BO", "TRENT.BO", "ADANITOTAL.BO", "LUPIN.BO", "CHOLAFIN.BO",
    "UBL.BO", "CANBK.BO", "BANKBARODA.BO", "ICICIGI.BO", "AUROPHARMA.BO", "PETRONET.BO",
    "TVSMOTOR.BO", "ADANIWILMAR.BO", "HAL.BO", "BEL.BO", "IDBI.BO", "ZOMATO.BO", "IRCTC.BO",
    "PAYTM.BO", "POLYCAB.BO", "APLLTD.BO", "INDHOTEL.BO", "BALKRISIND.BO", "MUTHOOTFIN.BO",
    "ATGL.BO", "PFC.BO", "ADANIGAS.BO", "MFSL.BO", "HONAUT.BO", "CROMPTON.BO", "DLF.BO",
    "INDIAMART.BO", "TATAPOWER.BO", "TORNTPOWER.BO", "MPHASIS.BO", "ABBOTINDIA.BO",
    "ASHOKLEY.BO", "CONCOR.BO", "GODREJPROP.BO", "IPCALAB.BO", "MINDTREE.BO", "LICHSGFIN.BO",
    "ASTRAL.BO", "HDFCAMC.BO", "PAGEIND.BO", "AUBANK.BO", "COFORGE.BO", "GLAXO.BO", "VOLTAS.BO",
    "MARICO.BO", "ESCORTS.BO", "SUNTV.BO", "SYNGENE.BO", "METROPOLIS.BO", "ALKEM.BO",
    "BATAINDIA.BO", "CUMMINSIND.BO", "RAMCOCEM.BO", "LALPATHLAB.BO", "GMRINFRA.BO",
    "BHARATFORG.BO", "JINDALSTEL.BO", "NAM-INDIA.BO", "DEEPAKNTR.BO", "ZYDUSLIFE.BO",
    "AARTIIND.BO", "COROMANDEL.BO", "MRF.BO", "GUJGASLTD.BO", "TATACHEM.BO", "SAIL.BO",
    "SUNDARMFIN.BO", "PERSISTENT.BO", "AMARAJABAT.BO", "WHIRLPOOL.BO", "GODREJIND.BO",
    "SUMICHEM.BO", "PIIND.BO", "JSWENERGY.BO", "ABB.BO", "HINDALCO.BO", "CANFINHOME.BO",
    "EXIDEIND.BO", "SRTRANSFIN.BO", "IDFCFIRSTB.BO", "SUPREMEIND.BO", "NIACL.BO",
    "TATACOMM.BO", "SCHAEFFLER.BO", "AIAENG.BO", "APOLLOTYRE.BO", "FSL.BO", "IEX.BO",
    "IRFC.BO", "JBCHEPHARM.BO", "JUBLINGREA.BO", "KAJARIACER.BO", "L&TFH.BO", "LINDEINDIA.BO",
    "M&MFIN.BO", "MFSL.BO", "NATIONALUM.BO", "NAVINFLUOR.BO", "OBEROIRLTY.BO", "PNB.BO",
    "POLYMED.BO", "PVR.BO", "RECLTD.BO", "SBICARD.BO", "SHRIRAMFIN.BO", "SONACOMS.BO",
    "STAR.BO", "SUNDRMFAST.BO", "TATAMTRDVR.BO", "THERMAX.BO", "TIINDIA.BO", "TORNTPOWER.BO",
    "TRENT.BO", "TRIDENT.BO", "TTKPRESTIG.BO", "UBL.BO", "UCOBANK.BO", "ULTRACEMCO.BO",
    "UNIONBANK.BO", "UPL.BO", "VBL.BO", "VEDL.BO", "VGUARD.BO", "VINATIORGA.BO", "VOLTAS.BO",
    "WHIRLPOOL.BO", "WIPRO.BO", "YESBANK.BO", "ZEEL.BO", "ZYDUSLIFE.BO"
]


# Streamlit select box for stock selection
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Slider to select prediction period
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data

def load_data(ticker) :

    data = yf.download (ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text ("Load data...")

data = load_data(selected_stock)

data_load_state.text("Loading data ... done !")

st.subheader('Raw data')
st.write(data.tail())
import plotly.graph_objects as go
def plot_raw_data():
    # Ensure Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Open'], 
        mode='lines', 
        name='Stock Open',
        line=dict(color='cyan')  # Use a visible color
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        mode='lines', 
        name='Stock Close',
        line=dict(color='yellow')  # Use a visible color
    ))

    fig.update_layout(
        title="Stock Price Time Series",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis_rangeslider_visible=True,
        template="plotly_dark"  # Ensures visibility in dark mode
    )

    st.plotly_chart(fig)

plot_raw_data()

# Fix MultiIndex issue if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Ensure column names are clean
data.columns = data.columns.map(lambda x: x.strip() if isinstance(x, str) else x)

# Select relevant columns
df_train = data[['Date', 'Close']].copy()

# Convert types
df_train['Date'] = pd.to_datetime(df_train['Date'], errors='coerce')
df_train['Close'] = pd.to_numeric(df_train['Close'], errors='coerce')

# Drop NaN values
df_train = df_train.dropna()

# Rename for Prophet
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Fit model
m = Prophet()
m.fit(df_train)

# Future predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display results
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)
st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

# About Section
st.sidebar.markdown("### About")
st.sidebar.info("""
**Aryan Kumar**  
                
💻 Aspiring Software Developer | AI & IoT Enthusiast  
📍 VIT Vellore | 8th Semester  
💼 Currently interning in AI Chatbots & IoT  
🔗 [LinkedIn](https://www.linkedin.com/in/aryan-kumar-36a7431aa/) | [GitHub](https://github.com/AryanKumar18)
""")

# About Section
st.sidebar.markdown("### Areas of Interest")
st.sidebar.info("""
**Aryan Kumar**  

• 📊 Tech Stack : 📌 Tech Stack: Python, Flask, React, PostgreSQL, Streamlit, AI/ML
• 🌐 Future Aspirations : 🚀 Looking for : Software Developer Roles in AI, IoT, or Full-Stack Development
• 📞 Contact: +91 7975527076
• 📧 Email: aryankumar.30.2003@gmail.com

