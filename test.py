from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
transactions = [['milk', 'bread'], ['bread', 'diaper', 'beer'], ['milk', 'diaper', 'bread']]
encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded, columns=encoder.columns_)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

G = nx.from_pandas_edgelist(rules, source='antecedents', target='consequents')
nx.draw(G, with_labels=True)
plt.show()