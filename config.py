"""
Config for personal finance dashboard
"""

# path to transactions csv with "date,name,amount" as header
TRANSACTIONS_FILE = 'transactions.csv'
# used to read all dates e.g. in transactions file
DATE_FORMAT = '%d.%m.%y'

# account balance in your currency
BALANCE = 5000
# date of when the account balance was observed
BALANCE_DATE = '01.01.17'

# similar names
GROUPS = {
    'living':  ['food', 'clothes', 'rent']
}

# rgb values of color scheme
COLOR_SCHEME = [
    (204,235,197), (188,128,189), (217,217,217),
    (252,205,229), (179,222,105), (253,180,98),
    (128,177,211), (251,128,114), (190,186,218),
    (255,255,179), (141,211,199)
]

