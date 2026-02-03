import pandas as pd
import numpy as np
import random

def generate_ecommerce_data(num_days=1000):
    print(" Starting Data Generation ")

    dates = pd.date_range(start="2023-01-01",periods=num_days)

    data = []

    for date in dates:
        day_of_week = date.dayofweek
        is_weekend = 1 if day_of_week >= 4 else 0

        # Competitor prices ary from $45 to $55
        competitor_price = round(random.uniform(45.0,55.0),2)

        # We experiement with a range of different prices
        our_price = round(random.uniform(40.0, 60.0), 2)

        demand = 50

        price_diff = competitor_price - our_price
        demand += (price_diff*3) # For every $1 cheaper we are, we sell 3 more units

        # Random factors that affect price
        noise = random.randint(-5,5)
        demand += noise

        units_sold = max(0, int(demand)) # Dont sell negative units

        data.append([date, is_weekend, competitor_price, our_price, units_sold])

    df = pd.DataFrame(data, columns=[
    'date', 
    'is_weekend', 
    'competitor_price', 
    'our_price', 
    'units_sold'
    ])

    print(f"Generated {len(df)} rows of data")
    print("Here are the first 5 rows:")
    print(df.head()) # .head() shows the top of the table

    df.to_csv("sales_data.csv", index=False)
    print(" Data saved to 'sales_data.csv' ")

if __name__ == "__main__":
    generate_ecommerce_data()
