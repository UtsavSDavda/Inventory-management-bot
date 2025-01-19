import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import random
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlxtend.frequent_patterns import apriori, association_rules
import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
import warnings
import en_core_web_sm
import os
import re 
from fuzzywuzzy import fuzz,process
from word2number import w2n
from flask import Flask, request, jsonify,render_template
import glob
from waitress import serve
import enchant
dictn = enchant.Dict("en_US")
app = Flask(__name__)

warnings.filterwarnings("ignore")
missing_flag = 0
missing_args=[{"Date":None},{"Brand":None},{"Time":None},{"Number":None},{"Category":None},{"Worker":None}]
history = []

forecast_times = ("day","days","month","months","year","years","periods","period","week","weeks")
forecast_words = ("forecast","forecasts","predict","anticipate","foresee","foretell","estimate","surmise","speculate","infer","expect","sales","future sale","sale","future sales","predict","restock","advance stock","safe stock","estimation","anticipation","anticipate",
                  "forecasting","quantity","required","needed")
forecast_nexts = ("next","coming","upcoming")
patterns_forecast = [
    [{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"IS_DIGIT": True},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"},{"IS_DIGIT": True},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"}],
    [{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"},{"IS_DIGIT": True},{"OP":"*"}],
    [{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"IS_DIGIT": True},{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"}],
    [{"OP":"*"},{"IS_DIGIT": True},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"}],
    [{"OP":"*"},{"IS_DIGIT": True},{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"},{"LOWER": {"IN": forecast_times}},{"ENT_TYPE":"BRAND"},{"OP":"*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}}, {"OP": "*"}, {"LOWER": {"IN": forecast_nexts}}, {"OP": "*"},{"IS_DIGIT": True}, {"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"LOWER": "for"},{"OP": "*"}, {"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_nexts}},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"LOWER": "for"}, {"OP": "*"},{"LOWER": {"IN": forecast_nexts}}, {"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"LOWER": {"IN": forecast_nexts}},{"OP": "*"}, {"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"IS_DIGIT": True}, {"OP": "*"},{"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"LOWER": {"IN": forecast_nexts}},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"LOWER": {"IN": forecast_nexts}},{"OP": "*"}, {"IS_DIGIT": True}, {"OP": "*"},{"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"LOWER": {"IN": forecast_nexts}}, {"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"IS_DIGIT": True}, {"OP": "*"},{"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_nexts}},{"OP": "*"},{"OP": "*"}],
    [{"OP": "*"},{"IS_DIGIT": True}, {"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"LOWER": {"IN": forecast_nexts}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"IS_DIGIT": True}, {"OP": "*"},{"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_nexts}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_nexts}}, {"OP": "*"},{"LOWER": "for"},{"OP": "*"}, {"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}}, {"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}, {"LOWER": "for"}, {"OP": "*"},{"LOWER": {"IN": forecast_nexts}},{"OP": "*"}, {"LOWER": {"IN": forecast_words}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"IS_DIGIT": True},{"OP": "*"}],
    [{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_words}},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_times}},{"OP": "*"}, {"LOWER": {"IN": forecast_words}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}}, {"OP": "*"},{"LOWER": {"IN": forecast_words}}, {"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}}, {"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}, {"LOWER": {"IN": forecast_words}},{"OP": "*"}],
    [{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"IS_DIGIT": True}, {"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER":  {"IN": forecast_words}},{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"}],
    [{"OP": "*"},{"LOWER":  {"IN": forecast_words}},{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"}],
    [{"OP": "*"},{"LOWER": {"IN": forecast_times}},{"OP": "*"},{"IS_DIGIT": True},{"OP": "*"},{"LOWER": {"IN": forecast_words}},{"OP": "*"}],
    [{"OP":"*"},{"LOWER": {"IN": forecast_times}},{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN": forecast_words}},{"OP":"*"},{"LOWER": {"IN": forecast_times}},{"OP":"*"}]
    ]

multi_mba_words = ["market", "basket","analysis","apriori","frequent","itemsets","association","rules","rule","itemsets","item","items","set","co-occur","frequently","bought","together","frequent","purchase","commonly"]
single_mba_words = ["apriori","itemsets","basket","co-occur","together","association","mba"]
patterns_mba = [[{"OP":"*"},{"LOWER": {"IN": multi_mba_words}},{"LOWER": {"IN": multi_mba_words}},{"OP":"*"}],
                [{"OP":"*"},{"LOWER": {"IN": multi_mba_words}},{"IS_DIGIT":True},{"LOWER": {"IN": multi_mba_words}},{"OP":"*"}],
                [{"OP":"*"},{"LOWER": {"IN": multi_mba_words}},{"LOWER": {"IN": multi_mba_words}},{"LOWER": {"IN": multi_mba_words}},{"OP":"*"}],
                [{"OP":"*"},{"LOWER": {"IN": single_mba_words}},{"OP":"*"},{"LOWER":"algorithm"},{"OP":"*"}],
                [{"OP":"*"},{"LOWER":"algorithm"},{"OP":"*"},{"LOWER": {"IN": single_mba_words}},{"OP":"*"}],
                ]
brand_performance_words = [
                ["best","most","highest","top","high","good","leading","biggest"],
                ["selling","performing","performance","selling","sold","performed","loved","liked","performers","performer","achiever"],
                ["brand","brands","product","products","comapnies","company"]
                                ]

lookup_prepositions = ["in","on","at","by","near","beside","next","behind","under","over","above",
                "below","around","between","inside","outside","beneath","across","within","throughout","much","many"]
lookup_words = ["worker","location","item","items","lookup","placed","quantity","number","located","pickup","picked","look","scan"]
lookup_objects = ["worker","location","item","date"]
lookup_actions = ["look","lookup","check","get"]
lookup_words_pick = ["pick","picks","picked","taken","obtained","moved","took"]
lookup_patterns = [
    [{"OP":"*"},{"LOWER":{"IN":lookup_prepositions}},{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"OP":"*"},{"LOWER":{"IN":lookup_prepositions}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"LOWER":{"IN":lookup_prepositions}},{"OP":"*"},{"LOWER":"date"},{"ENT_TYPE":"DATE"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"LOWER":{"IN":lookup_prepositions}},{"OP":"*"},{"ENT_TYPE":"DATE"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"ENT_TYPE":"DATE"},{"LOWER":{"IN":lookup_prepositions}},{"OP":"*"}],
    [{"OP":"*"},{"ENT_TYPE":"DATE"},{"LOWER":{"IN":lookup_words}},{"LOWER":{"IN":lookup_prepositions}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"worker"},{"LOWER":"id"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"worker"},{"LOWER":"by"},{"OP":"*"},{"LOWER":{"IN":lookup_words_pick}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words_pick}},{"OP":"*"},{"LOWER":"worker"},{"OP":"*"},{"LOWER":"by"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words_pick}},{"OP":"*"},{"LOWER":"by"},{"OP":"*"},{"LOWER":"worker"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"by"},{"OP":"*"},{"LOWER":"worker"},{"LOWER":{"IN":lookup_words_pick}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"by"},{"LOWER":{"IN":lookup_words_pick}},{"OP":"*"},{"LOWER":"worker"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"worker"},{"OP":"*"},{"LOWER":{"IN":lookup_words_pick}},{"OP":"*"},{"LOWER":"by"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"OP":"*"},{"LOWER":"row"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"OP":"*"},{"LOWER":"slot"},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"slot"},{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":"row"},{"OP":"*"},{"LOWER":{"IN":lookup_words}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_actions}},{"OP":"*"},{"LOWER":{"IN":lookup_objects}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER":{"IN":lookup_objects}},{"OP":"*"},{"LOWER":{"IN":lookup_actions}},{"OP":"*"}]
]

compare_sales_words = ["compare","contrast"]

patterns_compare = [[{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"LOWER":{"IN":compare_sales_words}},{"OP":"*"}],
                    [{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"LOWER":{"IN":compare_sales_words}},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"}],
                    [{"OP":"*"},{"LOWER":{"IN":compare_sales_words}},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"},{"ENT_TYPE":"BRAND"},{"OP":"*"}]
                    ]
ignored = []
listx = [forecast_nexts,forecast_times,forecast_words,compare_sales_words,lookup_words,lookup_words_pick,brand_performance_words[0],brand_performance_words[1],brand_performance_words[2]]
for i in range(len(listx)):
    for j in range(len(listx[i])):
        ignored.append((listx[i][j]).lower())

def lookup_byworkerid(id,pickdf):
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
        msgs.append("0")
        return msgs 
    for pick in pickup:
        msg = f"{id} picked up {pick[1]} {pick[-1]} of {pick[0]} from bin number {pick[4]} at {pick[6]} on {pick[5]}. Avaliable Units of {pick[0]}: {pick[2]}"
        msgs.append(msg)
    return msgs

def checkstockbyitem(item,invendf):
    isthere = 0
    row = 0
    for x in range(len(invendf['SKU'])):
        if invendf['SKU'][x] == item:
            isthere+=1
            row = x
            return invendf['QTY'][x]
    return -1

def lookup_bydate(date,pickdf):
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

def get_locationbyid(id,invendf):
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
        msg = "0"
        return msg
    
def getdatabylocation(loc,invendf):
    isfound = 0
    msg = []
    final = []
    for locs in invendf['LOCATION'].unique():
        if locs == loc:
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

def displayforecast(time_series_data,predicted_values,brandname):
    current_dir = os.getcwd()
    random_number = random.randint(1,1000)
    dir_path = os.path.join(current_dir, "static")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'figure_{timestamp}_{random_number}.jpg'
    filepath = os.path.join(dir_path,filename)
    fig, ax = plt.subplots()
    ax.plot(range(len(time_series_data)), time_series_data, marker='o', label='Actual Data')
    ax.plot(range(len(time_series_data), len(time_series_data) + len(predicted_values)), predicted_values, marker='o', label='Predicted Values')
    ax.set_title(brandname)
    ax.legend() 
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True)
    fig.savefig(os.path.join(dir_path, filename))
    plt.close(fig)

def forecast_sales(brandname,salesdf,periods=3):
    isfound = 0
    msgs = []
    for x in list(salesdf['Brand']):
        if x.lower() == brandname.lower():
            isfound+=1
    if isfound == 0:
        return f"No sales found for Brand {brandname}"
    df1 = salesdf[salesdf['Brand']==brandname]
    df1['Date'] = pd.to_datetime(df1['Date'],format="mixed")
    df1 = df1.sort_values(by="Date")
    quantity = list(df1['Quantity Sold'])
    dates = list(df1['Date'])
    forecasted_dates = []
    start_date = dates[-1]
    for x in range(periods):
        new_date = start_date + pd.Timedelta(days=1)
        start_date = start_date + pd.Timedelta(days=1)
        forecasted_dates.append(new_date.date())
    model = ExponentialSmoothing(quantity,trend="mul",seasonal="mul",seasonal_periods=4)
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    displayforecast(quantity,list(forecast),brandname)
    for cast in range(len(forecast)):
        msg = f"{round(forecast[cast])} Units of {brandname} to be sold at date {str(forecasted_dates[cast])} \n"
        msgs.append(msg)
    return msgs,forecast

def getquantitybybrand(brand_name,salesdf):
    invendata = pd.read_csv("inventorylist_appended.csv")
    invendf = pd.DataFrame(invendata)
    brands = salesdf.groupby('Brand')
    group = None
    current_stock = 0
    for brand, groupx in brands:
        if brand_name == brand:
            group = groupx
    uniqueitems = []
    for x in list(group['Item Number'].unique()):
        uniqueitems.append(x)
    for item in uniqueitems:
        current_stock += checkstockbyitem(item,invendf)
    return int(current_stock) 

def quantityrequired(brand_name,days,salesdf):
    msgs,forecast = forecast_sales(brand_name,salesdf=salesdf,periods=days)
    current_stock = getquantitybybrand(brand_name,salesdf)
    req = 0
    for x in range(len(forecast)):
        req+=forecast[x]
    if current_stock < int(req):
        message = f"Restock required of {round(req-current_stock)} Units to meet the forecasted demand of {int(req)} till the next {days} days."
    elif current_stock == req:
        message = f"The current stock {round(current_stock)} just meets the forecasted demand of {int(req)} for the next {days} days."
    else:
        message = f"The current stock {round(current_stock)} more than meets the forecasted demand of {int(req)} for the next {days} days."
    msgs.append(message)
    return msgs
    
def brand_performance(brand,salesdf):
    df1 = salesdf[salesdf["Brand"]==brand]
    sales = list(df1['Quantity Sold'])
    total_sales = 0
    diff = 0
    for x in sales:
        total_sales+=int(x)
    current_stock = getquantitybybrand(brand,salesdf)
    if current_stock > total_sales:
        diff = current_stock - total_sales
    return total_sales,((diff/current_stock))

def comparison_brand_performance(category,salesdf):
    df1 = salesdf[salesdf["Category"]==category]
    unique_brands = list(df1["Brand"].unique())
    msgs_sales_top = []
    msgs_stock = []
    performance = []
    for brand in unique_brands:
        performance.append((brand,brand_performance(brand,salesdf)))
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

def is_number(word):
    
    return any(char.isdigit() for char in word)
def is_number_word(word):
    try:
        w2n.word_to_num(word)
        return True
    except ValueError:
        return False
def replace_number_words(doc):
    new_tokens = []
    for token in doc:
        token_text = token.text

        if token_text.isdigit():
            
            new_tokens.append(token_text)
        elif is_number_word(token_text):
           
            try:
                number = w2n.word_to_num(token_text)
                new_tokens.append(str(number))

            except ValueError:
                
                new_tokens.append(token_text)
        else:

            new_tokens.append(token_text)
    
    return ' '.join(new_tokens)

def chat_forecast(doc):
    salesdata = pd.read_csv("category_sales.csv")
    salesdf = pd.DataFrame(salesdata)
    forecast_entities = [[],[],[],[]]
    isbrandfound = 0
    msgs = []
    amt = 1
    for token in doc:    
            if token.lower_ in forecast_times:
                forecast_entities[0].append(token.text)
            elif token.like_num == True:
                forecast_entities[1].append(int(token.text))
            elif token.lower_ in forecast_words:
                forecast_entities[2].append(token.text)
            else:
                forecast_entities[3].append(token.text)
    for i in range(4):
        forecast_entities[i] = list(set(forecast_entities[i]))
    if len(forecast_entities[0]) == 0:
        msgs.append("Please enter the time category ie. days, or months for example.")
        missing_flag = 1
        return msgs 
    if len(forecast_entities[3])==0:
        msgs.append("Please enter a BRAND name.")
        missing_flag = 1
        return msgs   
    time = forecast_entities[0][0]
    amt = 1 if (len(forecast_entities[1]) == 0) else forecast_entities[1][0]
    if time =="month" or time == "months":
        amt = amt*30
    elif time == "year" or time == "years":
        amt = amt*365
    elif time == "week" or time == "weeks":
        amt = amt*7
    else: 
        pass
    for brandx in range(len(forecast_entities[3])):
        for brand in list(salesdf['Brand'].unique()):
                if brand.lower() == forecast_entities[3][brandx].lower():
                    isbrandfound+=1
                    msg = quantityrequired(brand,amt,salesdf)
                    msgs.append(f"FORECAST FOR {brand}")
                    for i in range(len(msg)):
                        msgs.append(msg[i]) 
    if isbrandfound == 0:
        msgs.append("BRAND not found in our dataset.Check the spell again.")
        missing_flag = 1
    return msgs 

def chat_mba(doc):
    msgs = []
    numbers = []
    numberfound = 0
    for token in doc:
        if token.like_num == True:
            numbers.append(int(token.text))
            numberfound+=1
            break 
    if numberfound == 0:
        top_itemsets,top_rules = market_basket_analysis()
        msgs.append("Using default minimum length of itemset : 3, showing top 10 itemsets by support and top 10 association rules(most frequent).")
    else:
        top_itemsets,top_rules = market_basket_analysis(length_of_itemset=numbers[0])
        msgs.append(f"Using default minimum length of itemset : {numbers[0]}, showing top 10 itemsets by support and top 10 association rules(most frequent).")
    msgs.append("Here are the TOP ITEMSETS:")
    for i in range(len(top_itemsets)):
        msgs.append(top_itemsets[i])
    msgs.append("HERE ARE THE TOP RULES:")
    for i in range(len(top_rules)):
        msgs.append(top_rules[i])
    return msgs 

def chat_brand_performance(doc):
    salesdata = pd.read_csv("category_sales.csv")
    salesdf = pd.DataFrame(salesdata)
    msgs = []
    category = None 
    isfound = 0
    for token in doc:
        for x in list(salesdf['Category'].unique()):
            if x.lower() == token.lower_:
                isfound+=1
                msgs_sales_top,msgs_stock = comparison_brand_performance(x,salesdf)
                msgs.append("BRAND performance toppers by sales:")
                for i in range(len(msgs_sales_top)):
                    msgs.append(msgs_sales_top[i])
                msgs.append("BRAND performance toppers by sellout:")
                for i in range(len(msgs_stock)):
                    msgs.append(msgs_stock[i])
                break 
    if isfound == 0:
        msgs.append("The data is not present for the current request. Please enter a valid category.")
        missing_flag = 1
    return msgs 
def chat_inventory(doc):
    invendata = pd.read_csv("inventorylist_appended.csv")
    invendf = pd.DataFrame(invendata)
    pickdata = pd.read_csv("inventorypicklist_appended.csv")
    pickdf = pd.DataFrame(pickdata)
    modified_tokens = [token.text.replace('-', '/') if token.text == '-' else token.text for token in doc]
    modified_text = ''.join(modified_tokens)
    msgs = []
    isworker = 0
    isdate = 0
    isitem = 0
    isnumber = 0
    numbers = []
    atrow = []
    atslot = []
    dates = []
    id = []
    worker_pattern  = r'([Tt][Pp]\d+/\d+)'
    date_pattern = r'(\d{2})[\/](\d{2})[\/](\d{4})'
    regex = re.compile(date_pattern)
    match = regex.search(modified_text)
    regex2 = re.compile(worker_pattern)
    match2 = regex2.search(modified_text)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        dates.append([day,month,year])
        isdate+=1 
        msgs.append("LOOKING UP BY DATE \n")
        day = str(dates[0][0])
        month =str(dates[0][1])
        year = dates[0][2]
        if dates[0][0] < 10:
            day = "0"+day
        if dates[0][1] < 10:
            month = "0"+month 
        res = lookup_bydate(f"{day}-{month}-{year}",pickdf)
        for i in range(len(res)):
            msgs.append(res[i])
        return msgs 
    if match2:
        workerid = ''
        for char in match2.group(1):
            if char == '/':
                workerid+='-'
            elif char == 't':
                workerid+='T'
            elif char == 'p':
                workerid+='P'
            else:
                workerid+=char
        res = lookup_byworkerid(workerid,pickdf)
        if res == "0":
            msgs.append("0 records for the specified worker.")
            return msgs 
        for r in range(len(res)):
            msgs.append(res[r])
        return msgs 
    for token in doc:
        if token.lower_ == "row" or token.lower_ == "rows":
            atrow.append(token.i)
        elif token.lower_ == "slot" or token.lower_ == "slots":
            atslot.append(token.i) 
        elif token.like_num == True:
            numbers.append(int(token.text))
            isnumber+=1
        else:
            id.append(token.text)
    if len(atrow) == 0 and len(atslot) == 0:
        for i in id:
            res = get_locationbyid(i,invendf)
            if res!= "0":
                isitem+=1
                msgs.append(res)
                return msgs
        msgs.append("Not able to fetch more lookup details. Please be a little specific so I can get some idea.TIP: While entering date enter in dd/mm/YYYY format.")
        missing_flag = 1
        return msgs 
    else:
        if len(atrow) == 0:
            msgs.append("Please add row details. \n")
            missing_flag = 1
            return msgs
        elif len(atslot) == 0:
            msgs.append("Please add slot details. \n")
        elif len(numbers) < 2:
            msgs.append("Please enter both the values required for row and slot numbers respectively. \n")
            missing_flag = 1
            return msgs 
        else: 
            slotn = int(numbers[1])
            rown = int(numbers[0])
            locatn = f"Row {rown}, slot {slotn}"
            res = getdatabylocation(locatn,invendf)
            msgs.append(locatn)
            for i in range(len(res)):
                msgs.append(res[i])
            return msgs   

def compare_brands(brandlist,salesdf):
    msgs = []
    datas = []
    brands = []
    averages = []
    maxs = []
    mins = []
    sellouts = []
    sales = []
    random_number = random.randint(1,1000)
    for brand in brandlist:
        for b in salesdf['Brand'].unique():
            if b.lower() == brand.lower():
                perform = brand_performance(b,salesdf)
                datas.append((b,perform))
                sales.append((b,perform[0]))
                sellouts.append((b,perform[1]))
                brands.append(b)
    sortedbysellout = sorted(datas,key = lambda x: x[1][1])
    sortedbysales = sorted(datas,key = lambda x: x[1][0],reverse=True)
    for brand in brands:
        df1 = salesdf[salesdf['Brand']==brand]
        df1['Date'] = pd.to_datetime(df1['Date'],format="mixed")
        df1 = df1.sort_values(by="Date")
        avrage = df1['Quantity Sold'].mean()
        min= df1['Quantity Sold'].min()
        max = df1['Quantity Sold'].max()
        averages.append((brand,avrage))
        mins.append((brand,min))
        maxs.append((brand,max))
        quantity = list(df1['Quantity Sold'])
        dates = list(df1['Date'])
        plt.plot(range(len(quantity)),quantity,label=brand)
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.grid(True)
    dir_path = os.path.join(os.getcwd(), "static")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path,f'compare_{random_number}.jpg'))
    plt.close()
    averagesx = sorted(averages,key = lambda x: x[1])
    minsx = sorted(mins,key = lambda x: x[1])
    maxsx = sorted(maxs,key = lambda x: x[1],reverse=True)
    msgs.append("The brands ranked by AVERAGE SALES:")
    for i in range(len(averagesx)):
        msgs.append(f"Brand: {averagesx[i][0]}, Sales: {averagesx[i][1]}")
    msgs.append("The brands ranked by MINIMUM SALES:")
    for i in range(len(minsx)):
        msgs.append(f"Brand: {minsx[i][0]}, Minimum Sale: {minsx[i][1]}")
    msgs.append("The brands ranked by MAXIMUM SALES:")
    for i in range(len(maxsx)):
        msgs.append(f"Brand: {maxsx[i][0]}, Maximum Sale: {maxsx[i][1]}")
    msgs.append("The brands ranked by TOTAL SALES:")
    for i in range(len(sortedbysales)):
        msgs.append(f"{sortedbysales[i][0]}, Sales: {sortedbysales[i][1][0]}")
    msgs.append("The brands ranked by SELLOUT:")
    for i in range(len(sortedbysellout)):
        msgs.append(f"{sortedbysellout[i][0]}, Sellout score : {sortedbysellout[i][1][1]}")
    return msgs
def chat_compare(doc):
    brands = []
    msgs = []
    salesdata = pd.read_csv("category_sales.csv")
    salesdf = pd.DataFrame(salesdata)
    msgs.append("Called the function.")
    for token in doc:
        if token.ent_type_ == "BRAND":
            brands.append(token.text)
    msgs = compare_brands(brands,salesdf)
    return msgs

def fuzzymatch(label,patterns,doc):
    score = fuzz.token_sort_ratio(doc.text.lower(),patterns)
    return score

def get_image_filenames(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.jpg')]
    return image_files

def bot(user_input):
    files = glob.glob(os.path.join(os.getcwd()+'/static', '*.jpg'))
    for f in files:
        os.remove(f)
    salesdata = pd.read_csv("category_sales.csv")
    salesdf = pd.DataFrame(salesdata)
    brand_performance_categories = list((x.lower() for x in salesdf["Category"].unique()))
    patterns_brand_performance = [
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"LOWER": {"IN":brand_performance_words[1]}},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"LOWER": {"IN":brand_performance_words[2]}},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"LOWER": {"IN":brand_performance_words[0]}},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"LOWER": {"IN":brand_performance_words[1]}},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"LOWER": {"IN":brand_performance_words[0]}},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"LOWER": {"IN":brand_performance_words[2]}},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[0]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"}],
    [{"OP":"*"},{"LOWER": {"IN":brand_performance_words[2]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_words[1]}},{"OP":"*"},{"LOWER": {"IN":brand_performance_categories}},{"OP":"*"}]
    ]
    brand_patterns = []
    unique_brands = list(salesdf['Brand'].unique())
    for brand in unique_brands:
        brand_patterns.append({"label": "BRAND", "pattern": brand})
        brand_patterns.append({"label":"BRAND", "pattern":brand.lower()})
    nlp = en_core_web_sm.load()
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(brand_patterns)
    corrected_uin = []
    for word in unique_brands:
        ignored.append(word.lower())
        dictn.add_to_pwl(word.lower())
    for word in brand_performance_categories:
        ignored.append(word.lower())
        dictn.add_to_pwl(word.lower())
    for word in user_input.split():
        if (dictn.check(word)==False):
            sugestn = dictn.suggest(word)
            if sugestn:
                corrected_uin.append(sugestn[0])
            else:
                corrected_uin.append(word)
        else:
            corrected_uin.append(word)
    user_inputx = ' '.join(corrected_uin)
    print(user_inputx) 
    image_flag = 0 
    response = ""
    forecast_calls = 0
    inventory_calls = 0
    mba_calls = 0
    brand_calls = 0
    compare_calls = 0
    matcher = Matcher(nlp.vocab)
    docx = nlp(user_inputx)
    doc = nlp(replace_number_words(docx))
    intent = None 
    matcher.add("forecast",patterns_forecast)
    matcher.add("mba",patterns_mba)
    matcher.add("brand",patterns_brand_performance)
    matcher.add("lookup",lookup_patterns)
    matcher.add("compare",patterns_compare)
    matches = matcher(doc)
    if len(matches)==0:
        high_score = ["",-1]  
        for label in [("mba",patterns_mba),("forecast",patterns_forecast),("brand",patterns_brand_performance),("lookup",lookup_patterns)]:
            score = fuzzymatch(label[0],label[1],doc)
            if score > high_score[1]:
                high_score[0] = label[0]
                high_score[1] = score 
        print(str(high_score[0])+"-->>"+str(high_score[1]))
        if high_score[1] > 50:
            match_label = high_score[0]
        else:
            response = "We are not able to understand the query at the moment.Please try different methods, check for spelling errors if any. Sorry for the inconvenience."
            return response,image_flag
    for match_id, start, end in matches:
        span = doc[start:end]
        match_label = nlp.vocab.strings[match_id]
        if match_label == "forecast":
            if forecast_calls == 0:
                msgs = chat_forecast(doc)
                for i in range(len(msgs)):
                    response = response+"\n"+str(msgs[i])+"\n"
                forecast_calls+=1
                image_flag = 1 
                return response,image_flag
            else:
                pass 
        elif match_label == "mba":
            if mba_calls == 0:
                msgs = chat_mba(doc)
                mba_calls+=1
                for i in range(len(msgs)):
                    response = response+"\n"+str(msgs[i])+"\n"
                return response,image_flag 
            
            else:
                pass 
        elif match_label == "brand":
            if brand_calls == 0:
                msgs = chat_brand_performance(doc)
                for i in range(len(msgs)):
                    response = response+"\n"+str(msgs[i])+"\n"
                brand_calls+=1
                return response,image_flag
            else:
                pass 
        elif match_label == "lookup":
            if inventory_calls == 0:
                msgs = chat_inventory(doc) 
                for i in range(len(msgs)):
                    response = response+"\n"+str(msgs[i])+"\n"
                inventory_calls+=1
                return response,image_flag
        elif match_label == "compare":
            if compare_calls == 0:
                msgs = chat_compare(doc) 
                for i in range(len(msgs)):
                    response = response+"\n"+str(msgs[i])+"\n"
                compare_calls+=1
                image_flag = 1
                return response,image_flag
            else:
                pass
        else:
            return "none of the above",image_flag
                  
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/chat', methods=['POST'])
def chat():
        
        data = request.get_json()
        user_input = data['message']
        history.append("your query\n\n"+user_input+"\n")
        input_list = user_input.split()
        if len(input_list) == 1:
            if (input_list[0]).lower() == "history":
                bot_response = ""
                for x in history:
                    bot_response+=x 
                    bot_response+=" "
                image_flag = 0
            else: 
                bot_response,image_flag = bot(user_input)
                history.append("my response\n\n"+bot_response+"\n")
        else:
            bot_response,image_flag = bot(user_input)
            history.append("my response:\n\n"+bot_response+"\n")
        current_dir = os.getcwd()
        dir_path = os.path.join(current_dir, "static")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filenames = get_image_filenames(dir_path)
        return jsonify({"message":bot_response,"image_flag":image_flag,"images":filenames})
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080,threads=4)