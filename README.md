<h1>⚡ RevMax: AI Dynamic Pricing Engine</h1>

<h2>Revenue Optimization for E-Commerce</h2>
<p>A Deep Learning approach to finding the perfect price point.</p>

<h2>📑 Table of Contents</h2>
<ul>
  <li>Project Overview</li>
  <li>Target Use Case</li>
  <li>Architecture</li>
  <li>Installation & Setup</li>
  <li>How to Run</li>
  <li>Project Structure</li>
  <li>Features</li>
</ul>

<h2>📖 Project Overview</h2>
<p>
RevMax is an intelligent pricing agent designed for mid-range e-commerce products.
It utilizes a neural network to analyze market conditions (competitor pricing, day of the week, seasonality)
and predicts the optimal price point to maximize revenue.
</p>

<p>
Unlike simple rule-based systems (e.g., "always be $1 cheaper"), RevMax understands price elasticity.
It knows when to undercut competitors to drive volume and when to hold price to maximize margins.
</p>

<h2>🎯 Target Use Case</h2>

<h3>Product Category</h3>
<p>Mid-range consumer electronics, fashion, or home goods.</p>

<h3>Price Band</h3>
<p>$40.00 – $60.00.</p>

<h3>Market Dynamic</h3>
<p>Highly competitive environments where a $0.50 difference can significantly impact sales velocity.</p>

<h2>🏗️ Architecture</h2>
<p>The system consists of three distinct components:</p>

<h3>The Brain (PyTorch)</h3>
<p>A feed-forward neural network trained on historical sales data to predict demand curves.</p>

<h3>The Engine (FastAPI)</h3>
<p>A high-performance REST API that serves predictions and runs the optimization algorithms.</p>

<h3>The Control Center (Streamlit)</h3>
<p>An executive dashboard for visualizing real-time revenue curves and strategy insights.</p>

<h2>🛠️ Installation & Setup</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.8 or higher installed.</li>
</ul>

<h3>1. Clone the Repository</h3>
<pre><code>git clone https://github.com/yourusername/revmax-pricing.git
cd revmax-pricing
</code></pre>

<h3>2. Create a Virtual Environment</h3>
<p>It is highly recommended to use a virtual environment to manage dependencies.</p>

<pre><code># Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
</code></pre>

<h3>3. Install Dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<p>
(Note: Ensure you have pandas, numpy, torch, fastapi, uvicorn, streamlit, plotly, requests, and scikit-learn installed).
</p>

<h2>🚀 How to Run</h2>
<p>This project runs in three stages. You must run them in order for the first time.</p>

<h3>Step 1: Data Engineering</h3>
<p>Generate the synthetic dataset that simulates the market logic.</p>

<pre><code>python data_setup.py
</code></pre>
<p><strong>Output:</strong> Creates sales_data.csv</p>

<h3>Step 2: Model Training</h3>
<p>Train the neural network on the generated data.</p>

<pre><code>python train_model.py
</code></pre>
<p><strong>Output:</strong> Creates pricing_model.pth (weights) and scaler.save (normalization logic).</p>

<h3>Step 3: Launch the Backend (API)</h3>
<p>Start the FastAPI server. Keep this terminal window OPEN.</p>

<pre><code>uvicorn api:app --reload
</code></pre>

<p>Access API Docs at: http://127.0.0.1:8000/docs</p>

<h3>Step 4: Launch the Frontend (Dashboard)</h3>
<p>Open a new terminal, activate your virtual environment, and run:</p>

<pre><code>streamlit run dashboard.py
</code></pre>

<p>The dashboard will automatically open in your web browser.</p>

<h2>📂 Project Structure</h2>

<pre><code>revmax-pricing/
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
</code></pre>

<h2>📈 Features</h2>
<ul>
  <li><strong>Real-time Optimization:</strong> Suggests prices based on live competitor inputs.</li>
  <li><strong>Revenue Curve Visualization:</strong> See exactly how price changes affect projected revenue.</li>
  <li><strong>Strategic Insights:</strong> Explains why a price was chosen (e.g., "undercutting market median").</li>
  <li><strong>Docker Ready:</strong> (Optional) Can be easily containerized for cloud deployment.</li>
</ul>
