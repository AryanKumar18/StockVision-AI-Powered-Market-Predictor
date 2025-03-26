import pandas as pd
import streamlit as st
from datetime import date
import yfinance as yf
import plotly.graph_objects as go

# Define start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Prediction App")

# Stock selection
stocks = ["AAPL", "GOOG", "MSFT", "GME", "RELIANCE.NS"]
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Slider for prediction period
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Function to load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

# Display raw data
st.subheader("Raw data")
st.write(data.tail())

# Function to plot stock prices
def plot_raw_data():
    # Convert Date column to datetime format
    data["Date"] = pd.to_datetime(data["Date"])
    
    fig = go.Figure()

    # Stock Open Prices
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Open"].round(2),  # Round to 2 decimal places
        mode="lines",
        name="Stock Open",
        line=dict(color="cyan")  
    ))

    # Stock Close Prices
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Close"].round(2),  # Round to 2 decimal places
        mode="lines",
        name="Stock Close",
        line=dict(color="yellow")  
    ))

    # Update layout for better readability
    fig.update_layout(
        title="Stock Price Time Series",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis=dict(
            tickformat="%Y-%m-%d",  # Ensure proper date formatting
            tickangle=-45  # Rotate dates for better visibility
        ),
        yaxis=dict(
            tickformat=".2f"  # Display up to 2 decimal places
        ),
        xaxis_rangeslider_visible=True,
        template="plotly_dark"  # Ensure visibility in dark mode
    )

    st.plotly_chart(fig)

# Call function to plot the graph
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


st.sidebar.markdown("### Areas of Interest")
st.sidebar.info("""
**Aryan Kumar**  
                
🛠  Tech Stack 
📍  Future Aspirations  
📞  Contact: +91 7975527076  
📧  Email: aryankumar.30.2003@gmail.com
""")
