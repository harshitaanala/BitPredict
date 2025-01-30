import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import requests  # For fetching news
from prophet import Prophet

# Create the main window
root = tk.Tk()
root.title("Bitcoin Price Prediction Dashboard")
root.geometry("900x700")

# Title & Description
title_label = tk.Label(root, text="Bitcoin Price Prediction", font=("Arial", 18, "bold"))
title_label.pack(pady=10)
desc_label = tk.Label(root, text="Predict Bitcoin prices using Facebook Prophet", font=("Arial", 12))
desc_label.pack()

# User Inputs Section
frame = tk.Frame(root)
frame.pack(pady=20)

# Date Selection
date_label = tk.Label(frame, text="Prediction Period:")
date_label.grid(row=0, column=0, padx=5, pady=5)
days_var = tk.IntVar(value=30)
date_entry = ttk.Entry(frame, textvariable=days_var)
date_entry.grid(row=0, column=1, padx=5, pady=5)

# Changepoint Prior Scale Slider
changepoint_label = tk.Label(frame, text="Changepoint Prior Scale:")
changepoint_label.grid(row=1, column=0, padx=5, pady=5)
changepoint_var = tk.DoubleVar(value=0.05)
changepoint_slider = ttk.Scale(frame, from_=0.001, to=0.5, variable=changepoint_var, orient="horizontal")
changepoint_slider.grid(row=1, column=1, padx=5, pady=5)

# Seasonality Mode Dropdown
seasonality_label = tk.Label(frame, text="Seasonality Mode:")
seasonality_label.grid(row=2, column=0, padx=5, pady=5)
seasonality_var = tk.StringVar(value="additive")
seasonality_dropdown = ttk.Combobox(frame, textvariable=seasonality_var, values=["additive", "multiplicative"])
seasonality_dropdown.grid(row=2, column=1, padx=5, pady=5)

# Growth Type Dropdown
growth_label = tk.Label(frame, text="Growth Type:")
growth_label.grid(row=3, column=0, padx=5, pady=5)
growth_var = tk.StringVar(value="linear")
growth_dropdown = ttk.Combobox(frame, textvariable=growth_var, values=["linear", "logistic"])
growth_dropdown.grid(row=3, column=1, padx=5, pady=5)

# Fetch Data Button
def fetch_and_predict():
    # TODO: Load historical Bitcoin price data from a CSV or API
    bitcoin_df = pd.read_csv("dataset.csv\dataset.csv")  # Placeholder
    #bitcoin_df = None  # Replace this with actual data loading logic

    if bitcoin_df is None:
        result_label.config(text="Error: Load data before predicting!", fg="red")
        return

    # Preprocessing - Ensure 'ds' and 'y' columns exist
    bitcoin_df.rename(columns={"date": "ds", "price": "y"}, inplace=True)

    # Prophet Model
    model = Prophet(changepoint_prior_scale=changepoint_var.get(), seasonality_mode=seasonality_var.get(), growth=growth_var.get())
    model.fit(bitcoin_df)

    # Generate future dates
    future = model.make_future_dataframe(periods=days_var.get())

    # Make predictions
    forecast = model.predict(future)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bitcoin_df['ds'], bitcoin_df['y'], label="Actual Prices", color="blue")
    ax.plot(forecast['ds'], forecast['yhat'], label="Predicted Prices", color="red")
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="gray", alpha=0.2)
    ax.legend()

    # Embed plot in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    result_label.config(text="Prediction Successful!", fg="green")

predict_button = tk.Button(root, text="Predict", command=fetch_and_predict, bg="blue", fg="white", font=("Arial", 12, "bold"))
predict_button.pack(pady=10)

# Export to CSV Button
def export_csv():
    # TODO: Replace `forecast` with actual predicted dataframe
    forecast = None  # Replace with actual forecast DataFrame
    if forecast is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            forecast.to_csv(file_path, index=False)
            result_label.config(text="File Saved Successfully!", fg="green")

export_button = tk.Button(root, text="Export CSV", command=export_csv, bg="green", fg="white", font=("Arial", 12, "bold"))
export_button.pack(pady=10)

# News & Sentiment Analysis (Placeholder)
def fetch_news():
    # TODO: Replace with actual API key and logic
    api_key = "sk-proj-OSdY5gcdzh_A"  # Replace with your API key
    url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={api_key}"
    
    try:
        response = requests.get(url).json()
        articles = response.get("articles", [])
        news_text = "\n".join([f"- {article['title']}" for article in articles[:5]])  # Show top 5 news articles
    except Exception as e:
        news_text = "Error fetching news"

    news_label.config(text=news_text, justify="left", wraplength=600)

news_label = tk.Label(root, text="Fetching latest Bitcoin news...", font=("Arial", 10))
news_label.pack(pady=10)
news_button = tk.Button(root, text="Get Latest News", command=fetch_news, bg="orange", fg="black", font=("Arial", 12, "bold"))
news_button.pack(pady=5)

# Label for showing results/errors
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()
