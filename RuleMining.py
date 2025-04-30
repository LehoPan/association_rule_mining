import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt

# the data set consists of two separate spreadsheets they need to be read and then merged
data1 = pd.read_excel('./dataset/online_retail_II.xlsx', sheet_name='Year 2009-2010')
data2 = pd.read_excel('./dataset/online_retail_II.xlsx', sheet_name='Year 2010-2011')
raw_data = pd.concat([data1, data2], ignore_index=True)

# lets see the total shape of the data
print(f"Data shape: {raw_data.shape}")

# the dataset noted there are no missing values but checking here there seems to be a lot of missing columns
# print(dataset.isnull().sum())

# some of the stock codes contain only numbers so they will be accidentally processed as integers which will conflict with the other codes with letters
# so we are coverting all the values in the column into strings for consistency
raw_data['StockCode'] = raw_data['StockCode'].astype(str)

# the missing values are in the description and customerID columns, both of which we are dropping so we don't really mind.
dataset = raw_data.drop(columns=['Description', 'Customer ID', 'Quantity', 'Price', 'InvoiceDate', 'Country'])

# now they need to be grouped by invoice, to see which items were actually bought together in the same transaction
grouped = dataset.groupby('Invoice')['StockCode'].apply(list).reset_index()

# now that they are grouped we don't need the invoice column anymore
grouped.drop(columns=['Invoice'], inplace=True)


# gets rid of the metadata and titles, so it is just the sets of stock codes for each transaction
grouped_list = grouped['StockCode'].tolist()

# use the transaction encoder to turn it into a format for apriori rule association
encoder = TransactionEncoder()
encoded = encoder.fit(grouped_list).transform(grouped_list)
processed_data = pd.DataFrame(encoded, columns=encoder.columns_)

# extract the rules from the dataset
frequent_itemsets = apriori(processed_data, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

# print some sorted metrics by support, confidence, and lift. However what we want to focus on for this project is confidence specifically
print("Top 5 rulesets by support")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values("support", ascending=False).head(5))
print()

print("Top 5 rulesets by confidence")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values("confidence", ascending=False).head(5))
print()

print("Top 5 rulesets by lift")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values("lift", ascending=False).head(5))
print()

# shows a visualization of the rules extracted
G = nx.from_pandas_edgelist(rules, source='antecedents', target='consequents')
nx.draw(G, with_labels=True)
plt.show()

# function to remap the original product name back onto the stock code
def match_stock_ID(data, stock_code):
    name = data[data["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return name

# function that looks through rules with the desired product, and returns a list of items to bundle with it according to bundle_size
def bundle_rule(ruleset, product_id, bundle_size=1):
    # sort the rules by confidence since thats the metric that best suits our goal of recommending items to bundle together
    sorted = ruleset.sort_values("confidence", ascending=False)
    bundle = []
    for i, product in enumerate(sorted["antecedents"]):
        for j in list(product):
            if j == product_id:
                bundle.append(list(sorted.iloc[i]["consequents"])[0])

    return bundle[0:bundle_size]


# loop to allow user to pick multiple products and see what to bundle with them
while True:
    product_code = input("Please input a product code to see bundle suggestions: ")

    try:
        product_name = match_stock_ID(raw_data, product_code)
    except:
        print("Sorry we could not find any product with that code, please try again.")
        continue

    print(f"\nYou chose {product_name}.\n")
    
    bundle_size = int(input("How many items would you like to bundle with this item?: "))

    newbundle = bundle_rule(rules, product_code, bundle_size)
    print("Bundle recommendation:")
    for item in newbundle:
        product_name = match_stock_ID(raw_data, item)
        print(f"{item}: {product_name}")
    print()