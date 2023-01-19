#Business Problem
# The cart information of 3 different users is given below. Make the suggestion of the most suitable product for this cart information, using the rule of association.
# Product recommendations can be 1 or more than 1. Derive the rules from 2010-2011 Germany customers

#Variable Information
#InvoiceNo: Invoice number. A 6-digit number uniquely assigned to each transaction. If this code starts with the letter 'C', it indicates a cancellation.
#StockCode: Product (item) code. A 5-digit number uniquely assigned to each distinct product.
#Description: Product (item) name.
#Quantity: The quantities of each product (item) per transaction.
#InvoiceDate: The day and time when a transaction was generated.
#UnitPrice: Product price per unit in sterlin.
#CustomerID: A 5-digit number uniquely assigned to each customer.
#Country: The name of the country where a customer resides.

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

##################### Task 1 - Prepare data #####################################

#Step 1 - Load Dataset

df_ = pd.read_excel("WEEK_5/Ödevler/Bonus_öDEV/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
#Step 2: Drop the observation units whose StockCode is POST.
# (POST price added to each invoice does not represent the product.)
df = df[~df["StockCode"].str.contains("POST", na=False)]

# Step 3: Drop the observation units with null values.
df.dropna(inplace=True)

# Step 4: Extract the values containing C in Invoice from the data set.
# (C means the cancellation of the invoice.)
df = df[~df["Invoice"].str.contains("C", na=False)]

# Step 5: Filter out the observation units whose price and quantity is less than zero.
df = df[df["Price"] > 0]


# Step 6: Examine the outliers of the Price and Quantity variables, suppress them if necessary.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")


####################### Task 2: Generating Association Rules Through German Customers ######################
#Step 1: Create pivot table so as to use ARL

df_ger = df[df['Country'] == "Germany"]

create_invoice_product_df = df_ger.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)

create_invoice_product_df.iloc[0:5,0:5]

#Step 2: Create Rules

frequent_itemsets = apriori(create_invoice_product_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

######################################## Task 3 - Give an advice ################################
# Step 1: find a product name by utilizing stock_code
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_ger, 23215)


# Step 2: Make a product recommendation for 3 users using the arl_recommender function.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "SPACEBOY BEAKER", 3)
arl_recommender(rules, "DOLLY GIRL BEAKER", 3)
arl_recommender(rules, "LUNCH BAG APPLE DESIGN", 3)

