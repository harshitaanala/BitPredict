import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from prophet import Prophet
from datetime import datetime, timedelta

# Function to fetch Bitcoin historical data
def fetch_bitcoin_data():
    url = "https://api.coindesk.com/v1/bpi/historical/close.json"
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    response = requests.get(f"{url}?start={start_date}&end={end_date}")
    data = response.json()['bpi']
    
    df = pd.DataFrame(list(data.items()), columns=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# Function to train and predict using Prophet
def predict_prices(days):
    df = fetch_bitcoin_data()
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    return df, forecast

# Function to fetch latest Bitcoin news
def fetch_bitcoin_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "Bitcoin",
        "sortBy": "publishedAt",
        "apiKey": "cca1c767e31b4398b00a85e9469150d7",  # Replace with your API key
        "language": "en",
        "pageSize": 5
    }
    
    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])
    
    return [{"title": article["title"], "url": article["url"]} for article in articles]

# Streamlit UI
st.title("📈 BitPredict")
st.write("Predicts the Bitcoin prices using Facebook Prophet model along with displaying the latest crypto news.")

# Input field for prediction days
days = st.slider("Select prediction days:", min_value=30, max_value=365, value=90)

if st.button("Predict"):
    df, forecast = predict_prices(days)
    
    # Get the latest predicted price (for the last date in the forecast)
    latest_predicted_price = forecast['yhat'].iloc[-1]
    
    # Display the predicted price just above the graph
    st.subheader(f"Predicted Bitcoin Price: ${latest_predicted_price:,.2f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['ds'], df['y'], label="Actual Prices", color="blue")
    ax.plot(forecast['ds'], forecast['yhat'], label="Predicted Prices", color="red")
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="gray", alpha=0.2)
    ax.legend()
    
    st.pyplot(fig)

# Display latest Bitcoin news
st.subheader("📰 Latest Bitcoin News")
news = fetch_bitcoin_news()
for article in news:
    st.markdown(f"[{article['title']}]({article['url']})")
