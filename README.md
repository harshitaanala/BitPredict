# 📈 BitPredict
## Bitcoin Price Prediction and News Aggregator
BitPredict is a Bitcoin forecasting and news aggregation app that predicts future Bitcoin prices using the Facebook Prophet model and displays the latest Bitcoin news. The app features interactive price trend graphs and real-time news updates to help users stay informed about market trends.

## 🔍 Features
- ✔ Bitcoin Price Prediction using Prophet Model 📊
- ✔ Interactive Graphs for Historical & Forecasted Prices 📈
- ✔ Real-time Bitcoin News Fetching from News API 📰
- ✔ User-friendly UI built with Streamlit 🖥️
- ✔ Customizable Forecast Period (30 to 365 days) ⏳

## 🚀 Demo
Check out the live demo of the app: https://bitpredict-gswhhtnszuhjlwburavklg.streamlit.app/

## Screenshots
### Home page
![image](https://github.com/user-attachments/assets/af00f982-e1e7-46aa-a123-0fc77fde9c18)

### Prediction Page
![GUI2](https://github.com/user-attachments/assets/58b043fe-c4ce-4bcf-b020-98b36ed61aef)

## ⚡ Technologies Used
- Python 🐍 - Core Programming Language
- Streamlit 🌐 - Web Framework for Interactive UI
- Prophet 📈 - Time-Series Forecasting Model
- Matplotlib 📊 - Graph Plotting
- News API 📰 - Fetching Latest Bitcoin News

## 📥 Installation & Setup
### Prerequisites 
Ensure you have Python 3.x installed. You can check by running:

'''python --version
'''

### Clone the Repository

'''git clone https://github.com/harshitaanala/BitPredict.git'''

### Install Dependencies
Run the following command to install required Python libraries:

'''pip install -r requirements.txt
'''

### Set Up API Key for News
Get a free API key from NewsAPI.
Store it securely in Streamlit Secrets:

'''[general]
api_key = "add_your_api_key"
'''

### How to Run the Project
Run the following command to launch the app locally:

'''
streamlit run bitcoin_forecast.py
'''

## 🚀 Future Work
- Expand the app to support Ethereum, Dogecoin, and other major cryptocurrencies.
- Integrate WebSockets to provide real-time Bitcoin price updates instead of static predictions.
- Perform Natural Language Processing on news articles to analyze market sentiment (positive, neutral, or negative).
- Enable users to set alerts for when Bitcoin crosses a certain price threshold.

## Contact
📧 Email: harshitaanala@gmail.com  <br />
🔗 LinkedIn: https://www.linkedin.com/in/harshita-anala-ba8229228/


