import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define the class exactly as it was during training
class PricingModel(nn.Module):
    def __init__(self):
        super(PricingModel, self).__init__()
        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x
    
app = FastAPI()

print("Loading model and scaler...")

try:
    model = PricingModel()

    # Load trained mem
    model.load_state_dict(torch.load("pricing_model.pth"))
    model.eval() 
    scaler = joblib.load("scaler.save")
    print("System ready!")

except FileNotFoundError:
    print("ERROR: Run train_model.py first to generate the .pth and .save files!")


class PriceInput(BaseModel):
    competitor_price: float
    is_weekend: int # 0 for No, 1 for Yes


# API Endpoints
@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is Active. Use /docs for the interface."}

@app.post("/optimize")
def optimize_price(item: PriceInput):
    """
    1. Looks at the competitor price.
    2. Simulates prices from $20 to $70.
    3. Predicts sales for each price.
    4. Returns the price that gives the highest Revenue (Price * Units Sold).
    """
    best_price = 0
    best_revenue = 0
    best_predicted_sales = 0

    # np.arange creates a list: [20.0, 20.5, 21.0 ... 70.0]
    test_prices = np.arange(20.0, 70.0, 0.5)

    graph_data = []

    for i in test_prices:
        input_data = np.array([[item.competitor_price, i, item.is_weekend]])

        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad(): # Don't calculate gradients (saves memory)
            predicted_sales = model(input_tensor).item()

        sales = max(0, predicted_sales)
        revenue = i * sales

        graph_data.append({
            "price": float(i),
            "sales": float(sales),
            "revenue": float(revenue)
        })
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_price = i
            best_predicted_sales = sales

    return {
        "strategy": "Revenue Maximization",
        "competitor_price": item.competitor_price,
        "is_weekend": bool(item.is_weekend),
        "suggested_price": round(float(best_price), 2),
        "predicted_sales": round(float(best_predicted_sales), 2),
        "predicted_revenue": round(float(best_revenue), 2),
        "graph_data": graph_data # <--- We added this!
    }



