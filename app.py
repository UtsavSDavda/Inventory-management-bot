from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlxtend.frequent_patterns import apriori, association_rules
import spacy
from spacy.matcher import Matcher
import warnings
import en_core_web_sm

warnings.filterwarnings("ignore")

app = Flask(__name__)

salesdata = pd.read_csv("category_sales.csv")
salesdf = pd.DataFrame(salesdata)
invendata = pd.read_csv("inventorylist_appended.csv")
invendf = pd.DataFrame(invendata)
pickdata = pd.read_csv("inventorypicklist_appended.csv")
pickdf = pd.DataFrame(pickdata)

def lookup_byworkerid(id):
    pickup = []
    msgs = []
    length = 0
    foundid = 0
    for x in range(len(pickdf)):
        id1 = pickdf['ORDER #'][x]
        if id1.lower() == id.lower():
            length+=1
    if length > 1:
        df1 = pickdf[pickdf['ORDER #']==id]
        for x in df1.index:
            one = []
            one.append(df1['SKU'][x])
            one.append(df1['PICK QTY'][x])
            one.append(df1['QTY AVAILABLE'][x])
            one.append(df1['ITEM DESCRIPTION'][x])
            one.append(df1['BIN #'][x])
            one.append(df1['DATE'][x])
            one.append(df1['LOCATION'][x])
            if df1['UNIT'][x] == "Each":
                one.append("UNITS") if df1['PICK QTY'][x] > 1 else one.append("UNIT")
            else:
                one.append("BOXES") if df1['PICK QTY'][x] > 1 else one.append("BOX")
            pickup.append(one)       
    else:
        msgs.append("The ID is not found in the data")
        return msgs 
    for pick in pickup:
        msg = f"{id} picked up {pick[1]} {pick[-1]} of {pick[0]} from bin number {pick[4]} at {pick[6]} on {pick[5]}. Avaliable Units of {pick[0]}: {pick[2]}"
        msgs.append(msg)
    return msgs

def checkstockbyitem(item):
    isthere = 0
    row = 0
    for x in range(len(invendf['SKU'])):
        if invendf['SKU'][x] == item:
            isthere+=1
            row = x
            return invendf['QTY'][x]
    return -1

def lookup_bydate(date):
    isthere = 0
    pickup = []
    msgs= []
    for x in pickdf['DATE']:
        if x == date:
            isthere+=1
    if isthere==0:
        msgs.append("No units found")
        return msgs
    df1 = pickdf[pickdf['DATE']==date]
    if len(df1['DATE'])>1:
        for x in df1.index:
            one = []
            one.append(df1['SKU'][x])
            one.append(df1['PICK QTY'][x])
            one.append(df1['QTY AVAILABLE'][x])
            one.append(df1['ITEM DESCRIPTION'][x])
            one.append(df1['BIN #'][x])
            one.append(df1['ORDER #'][x])
            one.append(df1['LOCATION'][x])
            if df1['UNIT'][x] == "Each":
                one.append("UNITS") if df1['PICK QTY'][x] > 1 else one.append("UNIT")
            else:
                one.append("BOXES") if df1['PICK QTY'][x] > 1 else one.append("BOX")
            pickup.append(one) 
    else:
        one = []
        one.append(df1['SKU'][0])
        one.append(df1['PICK QTY'][0])
        one.append(df1['QTY AVAILABLE'][0])
        one.append(df1['ITEM DESCRIPTION'][0])
        one.append(df1['BIN #'][0])
        one.append(df1['ORDER #'][0])
        one.append(df1['LOCATION'][0])
        if pickdf['UNIT'][x] == "Each":
            one.append("UNITS") if df1['PICK QTY'][x] > 1 else one.append("UNIT")
        else:
            one.append("BOXES") if df1['PICK QTY'][x] > 1 else one.append("BOX")
        pickup.append(one)
    for pick in pickup:
        msg = f"{pick[5]} picked up {pick[1]} {pick[-1]} of {pick[0]} from bin number {pick[4]} at {pick[6]} on {date}. Avaliable Units of {pick[0]}: {pick[2]}"
        msgs.append(msg)
    return msgs

def get_locationbyid(id):
    find = 0
    for x in range(len(invendf['SKU'])):
        if invendf['SKU'][x] == id:
            find+=1
            sku = invendf['SKU'][x]
            location = invendf['LOCATION'][x]
            binno = invendf['BIN #'][x]
            qty = invendf['QTY'][x]
            msg = f"{qty} units of the item {sku} found at {location} with bin number {binno}."
    if find > 0:
        return msg
    else:
        msg = "The item is not found in inventory"
        return msg

def getdatabylocation(loc):
    isfound = 0
    msg = []
    final = []
    for locs in invendf['LOCATION'].unique():
        if locs.lower() == loc.lower():
            isfound+=1
            df1 = invendf[invendf['LOCATION']==locs]
            break 
    if isfound == 0:
        msg.append("Location not found.")
        return msg
    for x in df1.index:
        sku = df1['SKU'][x]
        binno = df1['BIN #'][x]
        qty = df1['QTY'][x]
        in_value = df1['INVENTORY VALUE'][x]
        final.append([sku,binno,qty,in_value])
    for f in final:
        msg.append(f"{f[2]} units of {f[0]} found at bin number {f[1]} at the location {loc} worth {f[3]}")
    return msg

def sales(brandname):
    sales = []
    isfound = 0
    for x in list(salesdf['Brand'].unique()):
        if str(x).lower() == brandname.lower():
            isfound+=1
            break
    if isfound == 0:
        sales.append("Brand Not Found.")
        return sales 
    df1 = salesdf[salesdf['Brand']==x]
    for i in df1.index:
        sales.append(df1["Quantity Sold"][i])
    return sales

def displayforecast(time_series_data, predicted_values, file_path='static/forecast_plot.png'):
   
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(time_series_data)), time_series_data, marker='o', label='Actual Data')
    plt.plot(range(len(time_series_data), len(time_series_data) + len(predicted_values)), predicted_values, marker='o', label='Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data with Exponential Smoothing Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
   

    
def forecast_sales(brandname,periods=3):
    isfound = 0
    msgs = []
    brand = None
    for x in list(salesdf['Brand']):
        if x.lower() == brandname.lower():
            isfound+=1
            brand = x
    if isfound == 0:
        return f"No sales found for Brand {brandname}"
    df1 = salesdf[salesdf['Brand']==brand]
    df1['Date'] = pd.to_datetime(df1['Date'],format="mixed")
    df1 = df1.sort_values(by="Date")
    quantity = list(df1['Quantity Sold'])
    dates = list(df1['Date'])
    forecasted_dates = []
    start_date = dates[-1]
    for x in range(periods):
        new_date = start_date + pd.Timedelta(days=1)
        start_date = start_date + pd.Timedelta(days=1)
        forecasted_dates.append(new_date)
    model = ExponentialSmoothing(quantity,trend="mul",seasonal="mul",seasonal_periods=4)
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    displayforecast(quantity,list(forecast))
    for cast in range(len(forecast)):
        msg = f"{forecast[cast]} Units to be sold at date {str(forecasted_dates[cast])}"
        msgs.append(msg)
    return msgs,forecast

def getquantitybybrand(brand_name):
    brands = salesdf.groupby('Brand')
    group = None
    current_stock = 0
    for brand, groupx in brands:
        if brand_name.lower() == brand.lower():
            group = groupx
    uniqueitems = []
    for x in list(group['Item Number'].unique()):
        uniqueitems.append(x)
    for item in uniqueitems:
        current_stock += checkstockbyitem(item)
    return int(current_stock) 

def quantityrequired(brand_name,days):
    msgs,forecast = forecast_sales(brand_name,days)
    current_stock = getquantitybybrand(brand_name)
    req = 0
    for x in range(len(forecast)):
        req+=forecast[x]
    if current_stock < int(req):
        message = f"Restock required of {req-current_stock} Units to meet the forecasted demand of {int(req)} till the next {days} days."
    elif current_stock == req:
        message = f"The current stock {current_stock} just meets the forecasted demand of {int(req)} for the next {days} days."
    else:
        message = f"The current stock {current_stock} more than meets the forecasted demand of {int(req)} for the next {days} days."
    return message
    
def brand_performance(brand):
    df1 = salesdf[salesdf["Brand"]==brand]
    sales = list(df1['Quantity Sold'])
    total_sales = 0
    diff = 0
    for x in sales:
        total_sales+=int(x)
    current_stock = getquantitybybrand(brand)
    if current_stock > total_sales:
        diff = current_stock - total_sales
    return total_sales,((diff/current_stock)*100)

def comparison_brand_performance(category):
    df1 = salesdf[salesdf["Category"]==category]
    unique_brands = list(df1["Brand"].unique())
    msgs_sales_top = []
    msgs_stock = []
    performance = []
    for brand in unique_brands:
        performance.append((brand,brand_performance(brand)))
    sortedbysales = list(sorted(performance,key = lambda x: x[1][0],reverse=True))
    sortedbyretention = list(sorted(performance,key = lambda x: x[1][1]))
    topsales = sortedbysales[0:5]
    bottomsales = sortedbysales[-1:-6:-1]
    range_sorted = len(sortedbysales) if (len(sortedbysales) < 5) else 5
    range_retention = len(sortedbyretention) if (len(sortedbyretention) < 5) else 5
    for x in range(range_sorted):
        msg = f"{sortedbysales[x][0]}, sales: {sortedbysales[x][1][0]}."
        msgs_sales_top.append(msg)
    for x in range(range_retention):
        msg = f"{sortedbyretention[x][0]}, retention: {sortedbyretention[x][1][0]}"
        msgs_stock.append(msg)
    return msgs_sales_top, msgs_stock

def displayfrequentitemsets(itemsets,top_n,min_length):
    sentences = []
    filtered_itemsets = itemsets[itemsets['itemsets'].apply(lambda x: len(x) >= min_length)]
    top_itemsets = filtered_itemsets.nlargest(top_n, 'support')
    for _, row in top_itemsets.iterrows():
        items = ', '.join(list(row['itemsets']))
        sentence = f"The itemset '{items}' appears in {row['support'] * 100:.2f}% of transactions."
        sentences.append(sentence)
    return sentences
def displayassociationrules(rules,top_n):
    sentences = []
    top_rules = rules.nlargest(top_n, 'confidence')
    for _, row in top_rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        sentence = (f"If a transaction contains ({antecedents}), "
                    f"it is likely to also contain ({consequents}) with "
                    f"a support of {row['support'] * 100:.2f}%, "
                    f"confidence of {row['confidence'] * 100:.2f}%, "
                    f"and lift of {row['lift']:.2f}.")
        sentences.append(sentence)
    return sentences

def market_basket_analysis(length_of_itemset=3,default_top_itemset=10,default_top_rule = 10):
    basket_data = pd.read_csv('basket_analysis.csv')
    basketdf = pd.DataFrame(basket_data) 
    basketdf.rename(columns={'Unnamed: 0': 'Transaction ID'}, inplace=True)
    basketdf.set_index('Transaction ID',inplace=True)
    basketdf = basketdf.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basketdf, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    top_itemsets = displayfrequentitemsets(frequent_itemsets,default_top_itemset,length_of_itemset)
    top_rules = displayassociationrules(rules,default_top_rule)
    return top_itemsets,top_rules

def convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    else:
        return data
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lookup', methods=['POST'])
def lookup():
    lookup_type = request.json.get('lookupType')
    msgs = []  # Initialize msgs
    if lookup_type == '1':
        worker_id = request.json.get('workerId')
        msgs = lookup_byworkerid(worker_id)
    elif lookup_type == '2':
        pickup_date = request.json.get('pickupDate')
        msgs = lookup_bydate(pickup_date)
    elif lookup_type == '3':
        item_id = request.json.get('itemId')
        msgs = [get_locationbyid(item_id)]  # Ensure to wrap single message in list
    elif lookup_type == '4':
        bin_location = request.json.get('binLocation')
        msgs = getdatabylocation(bin_location)
    elif lookup_type == '5':
        brand_name = request.json.get('brandName')
        qty = getquantitybybrand(brand_name)
        msgs = [f"The current stock for the brand {brand_name} is {qty}."]
    else:
        msgs = ["Invalid lookup type."]  # Handle invalid cases
    return jsonify(msgs)


@app.route('/analysis', methods=['POST'])
def analysis():
    analysis_type = request.json.get('analysisType')
    serializable_msgs =[] 
    
    if analysis_type == '1':
        brand_name = request.json.get('brandName')
        msgs = sales(brand_name)
        serializable_msgs = convert_to_serializable(msgs)
    elif analysis_type == '2':
        category_name = request.json.get('categoryName')
        msgs_sales_top, msgs_stock = comparison_brand_performance(category_name)
        msgs = msgs_sales_top + msgs_stock
        serializable_msgs = convert_to_serializable(msgs)
    elif analysis_type == '3':
        itemset_length = int(request.json.get('itemsetLength'))
        top_itemsets = int(request.json.get('topItemsets'))
        top_rules = int(request.json.get('topRules'))
        msgs_itemsets, msgs_rules = market_basket_analysis(itemset_length, top_itemsets, top_rules)
        msgs = msgs_itemsets + msgs_rules
        serializable_msgs = convert_to_serializable(msgs)
    else:
        serializable_msgs = ["Invalid analysis type."]  # Handle invalid cases
    return jsonify(serializable_msgs)


@app.route('/prediction', methods=['POST'])
def prediction():
    prediction_type = request.json.get('predictionType')
    brand_name = request.json.get('brandName')
    days = request.json.get('days')
    msgs = []  
    if prediction_type == '1':
        msgs, _ = forecast_sales(brand_name, days)
    elif prediction_type == '2':
        msgs = [quantityrequired(brand_name, days)]
    else:
        msgs = ["Invalid prediction type."]  # Handle invalid cases
    return jsonify(msgs)

if __name__ == '__main__':
    app.run(debug=True)