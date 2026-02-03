<div align="center">

⚡ RevMax: AI Dynamic Pricing Engine

Revenue Optimization for E-Commerce > A Deep Learning approach to finding the perfect price point.

</div>

📑 Table of Contents

Project Overview

Architecture

Installation & Setup

How to Run

Project Structure

Features

📖 Project Overview

RevMax is an intelligent pricing agent designed for mid-range e-commerce products. It utilizes a neural network to analyze market conditions (competitor pricing, day of the week, seasonality) and predicts the optimal price point to maximize revenue.

Unlike simple rule-based systems (e.g., "always be $1 cheaper"), RevMax understands price elasticity. It knows when to undercut competitors to drive volume and when to hold price to maximize margins.

🎯 Target Use Case

This software is specifically tuned for:

Product Category: Mid-range consumer electronics, fashion, or home goods.

Price Band: $40.00 – $60.00.

Market Dynamic: Highly competitive environments where a $0.50 difference can significantly impact sales velocity.

🏗️ Architecture

The system consists of three distinct components:

The Brain (PyTorch): A feed-forward neural network trained on historical sales data to predict demand curves.

The Engine (FastAPI): A high-performance REST API that serves predictions and runs the optimization algorithms.

The Control Center (Streamlit): An executive dashboard for visualizing real-time revenue curves and strategy insights.

🛠️ Installation & Setup

Prerequisites

Python 3.8 or higher installed.

1. Clone the Repository

git clone [https://github.com/yourusername/revmax-pricing.git](https://github.com/yourusername/revmax-pricing.git)
cd revmax-pricing


2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


(Note: Ensure you have pandas, numpy, torch, fastapi, uvicorn, streamlit, plotly, requests, and scikit-learn installed).

🚀 How to Run

This project runs in three stages. You must run them in order for the first time.

Step 1: Data Engineering

Generate the synthetic dataset that simulates the market logic.

python data_setup.py


Output: Creates sales_data.csv

Step 2: Model Training

Train the neural network on the generated data.

python train_model.py


Output: Creates pricing_model.pth (weights) and scaler.save (normalization logic).

Step 3: Launch the Backend (API)

Start the FastAPI server. Keep this terminal window OPEN.

uvicorn api:app --reload


Access API Docs at: https://www.google.com/search?q=http://127.0.0.1:8000/docs

Step 4: Launch the Frontend (Dashboard)

Open a new terminal, activate your virtual environment, and run:

streamlit run dashboard.py


The dashboard will automatically open in your web browser.

📂 Project Structure

revmax-pricing/
│
├── api.py             # FastAPI server (The Deployment)
├── dashboard.py       # Streamlit Dashboard (The Frontend)
├── data_setup.py      # Data generation script (The Data Engineering)
├── train_model.py     # PyTorch training script (The AI)
│
├── requirements.txt   # List of dependencies
├── sales_data.csv     # (Generated) Synthetic training data
├── pricing_model.pth  # (Generated) Saved model weights
└── scaler.save        # (Generated) Saved data scaler


📈 Features

Real-time Optimization: Suggests prices based on live competitor inputs.

Revenue Curve Visualization: See exactly how price changes affect projected revenue.

Strategic Insights: Explains why a price was chosen (e.g., "undercutting market median").

Docker Ready: (Optional) Can be easily containerized for cloud deployment.
