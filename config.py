"""
Config for personal finance dashboard
"""

import pandas as pd

# path to transactions csv with "date,name,amount" as header
TRANSACTIONS_FILE = 'transactions.csv'
# used to read all dates e.g. in transactions file
DATE_FORMAT = '%d.%m.%y'

# account balance in your currency
BALANCE = 5000
# date of when the account balance was observed
BALANCE_DATE = '31.12.17'

# read transactions
TRANSACTIONS = pd.read_csv(TRANSACTIONS_FILE)
TRANSACTIONS['date'] = pd.to_datetime(TRANSACTIONS['date'], format=DATE_FORMAT)
TRANSACTIONS = TRANSACTIONS[TRANSACTIONS['date'] > BALANCE_DATE]

# color scheme
qual_set3 = ['rgb(204,235,197)', 'rgb(188,128,189)', 'rgb(217,217,217)',
             'rgb(252,205,229)', 'rgb(179,222,105)', 'rgb(253,180,98)',
             'rgb(128,177,211)', 'rgb(251,128,114)', 'rgb(190,186,218)',
             'rgb(255,255,179)', 'rgb(141,211,199)']

names = TRANSACTIONS['name'].unique()
COLORS = {name: qual_set3[i % len(qual_set3)] for i, name in enumerate(names)}
COLORS['EXP'] = '#d1695c' # expense color
COLORS['INC'] = '#5da080' # income color


