from config import BALANCE, BALANCE_DATE, GROUPS, COLOR_SCHEME, TRANSACTIONS_FILE, DATE_FORMAT

from subprocess import Popen
import numpy as np
import pandas as pd
from typing import List
from pyramid.arima import auto_arima

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components import Th, Thead, Td, Div, H1, Table, Tr, Span
import plotly
from plotly.graph_objs import Bar, Layout, Figure, Pie, Scatter

TRANSACTIONS = pd.read_csv(TRANSACTIONS_FILE)
TRANSACTIONS['date'] = pd.to_datetime(TRANSACTIONS['date'], format=DATE_FORMAT)
TRANSACTIONS = TRANSACTIONS[TRANSACTIONS['date'] > BALANCE_DATE]

def colors_and_names():
    # every color extends to a range between 0.6 and 0.95 of its brightness
    color_steps = np.linspace(0.9, 0.7, num=max(map(len, GROUPS.values())) + 1)
    colors = [[[int(v * (256/max(rgb)) * cs) for v in rgb] for cs in color_steps]
               for rgb in COLOR_SCHEME]

    res = {}
    for i, (group, names) in enumerate(GROUPS.items()):
        r,g,b = colors[i % len(colors)][0]
        res[group] = (r,g,b,1.0)
        for j, name in enumerate(names,1):
            if name not in res:
                r,g,b = colors[i % len(colors)][j]
                res[name] = (r,g,b,1.0)

    # names without groups
    left_names = set(TRANSACTIONS['name']).difference(res.keys())
    for i, name in enumerate(left_names, len(GROUPS.keys())):
        r,g,b = colors[i % len(colors)][0]
        res[name] = (r,g,b,1.0)

    names = [n for n in res.keys() if n in TRANSACTIONS['name'].unique()]
    res['EXP'] = (209,105,92,1.0) # expense color
    res['INC'] = (93,160,128,1.0) # income color
    return res, names


def color(name: str, alpha=None, values=False):
    r,g,b,a = COLORS[name]
    new_alpha = a if alpha is None else min(1,max(0,alpha))
    if values:
        return (r,g,b,new_alpha)
    else:
        return f'rgba({r},{g},{b},{new_alpha})'


COLORS, NAMES = colors_and_names()


def transactions_monthly() -> pd.DataFrame:
    """
    Aggregate transactions to monthly table
    The indices are months and columns are transaction names
    """

    monthly = TRANSACTIONS.groupby([TRANSACTIONS['date'].dt.year,
                                    TRANSACTIONS['date'].dt.month,
                                    TRANSACTIONS['name']]).sum()
    monthly.index = pd.MultiIndex.from_tuples([(pd.datetime(y,m,1), name)
                                               for y,m,name in monthly.index])
    monthly = monthly['amount'].unstack().fillna(0)

    return monthly


def monthly_html_table() -> Div:
    """
    Build pretty monthly transaction table with names as indices and months as columns
    """
    df = transactions_monthly()
    df.index = [s.strftime('%b %y') for s in df.index]
    expenses = [n for n in NAMES if n in df.columns and df[n].sum() <= 0]
    income = [n for n in NAMES if n in df.columns and df[n].sum() > 0]
    df = df[expenses + income].T

    curr = ' {:,.2f}'.format
    perc = '{:,.2%}'.format

    def circle(color):
        css = {'width': '14px', 'height': '14px', 'background': color,
               'display': 'inline-block', 'margin': '0 8px 0 0', 'padding': 0}
        return Div(style=css)

    header_row = Thead(
        [Th('Name', scope='col')] + [Th(col, scope='col') for col in df.columns],
        className='thead-light'
    )

    # name rows
    name_rows = []
    for cat, row in df.iterrows():
        cells = [Th([circle(color(cat)), cat], scope='row', style={'background': '#fafafa'})]
        for cell in row:
            style = {'color': color('INC') if cell >= 0 else color('EXP'),
                     'text-align': 'right'}
            cells.append(Td(curr(cell or '') if cell != 0 else '', style=style))
        name_rows.append(Tr(cells))

    # growth row
    balances = list(df.sum().cumsum() + BALANCE)
    saving_rel = np.divide(balances, [BALANCE] + balances[:-1]) - 1
    saving_abs = np.array(balances) - np.array([BALANCE] + balances[:-1])

    growth_row = [Th('GROWTH', style={'background': '#f5f5f5', 'border-top': 'solid 2px #CCC'})]
    for sr, sa in zip(saving_rel, saving_abs):
        sa = Span(curr(sa), style={'color': color('INC') if sr >= 0 else color('EXP'), 'font-weight': 'bold'})
        sr = Span(f'{perc(sr)}', style={'color': color('INC', alpha=0.8) if sr >= 0 else color('EXP', alpha=0.8)})
        sep = Span('  |  ', style={'color': '#CCC'})
        growth_row.append(Td([sr, sep, sa], style={'border-top': 'solid 2px #CCC', 'text-align': 'right'}))
    growth_row = Tr(growth_row)


    # total row
    total_row = [Th('TOTAL', style={'background': '#f5f5f5'})]
    for exp, inc in zip(df.loc[expenses].sum(), df.loc[income].sum()):
        sexp = Span(curr(exp), style={'color': color('EXP'), 'font-weight': 'bold'})
        sinc = Span(curr(inc), style={'color': color('INC'), 'font-weight': 'bold'})
        sep = Span('  |  ', style={'color': '#CCC'})
        total_row.append(Td([sexp, sep, sinc],
                            style={'text-align': 'right'}))
    total_row = Tr(total_row)

    # balance row
    balance_row = [Th('BALANCE', style={'background': '#f5f5f5'})]
    for diff in df.sum().cumsum() + BALANCE:
        balance_row.append(Td(curr(diff), style={'font-weight': 'bold', 'text-align': 'right'}))
    balance_row = Tr(balance_row)
        
    all_rows = [header_row] + name_rows + [growth_row] + [total_row] + [balance_row]

    return Div(Table(all_rows, className='table table-bordered table-sm'),
               className='table-responsive-sm')


def forecast_traces(balances, weeks=52) -> List[Scatter]:
    """
    Forecast next weeks based on balance history with ARIMA
    """

    last_balance_date = balances.index.max()
    last_balance = balances.loc[last_balance_date]
    X_forecast = list(pd.date_range(last_balance_date, periods=weeks, freq='W'))

    model = auto_arima(
        balances, trend=[1, 1],
        error_action='ignore', suppress_warnings=True
    )
    forecast = model.predict(n_periods=weeks)

    # error estimation
    model_error = np.std(model.resid())
    sampling_error = np.sqrt(balances.var() / len(balances))
    forecast_error = 2 * np.sqrt(model_error**2 + sampling_error**2)

    forecast_upper = forecast + forecast_error
    forecast_lower = forecast - forecast_error
    bad_forecast = np.min(forecast) < balances.loc[last_balance_date]

    balances_forecast = Scatter(
        name='forecast', x=X_forecast, y=[last_balance] + list(forecast),
        mode='lines',
        line={'dash': 'dash', 'color': color('EXP') if bad_forecast else color('INC')},
        opacity=0.8
    )
    balances_forecast_upper = Scatter(
        name='', x=X_forecast, y=[last_balance] + list(forecast_upper),
        fill='tonexty', fillcolor=(color('EXP' if bad_forecast else 'INC', alpha=0.2)),
        line={'color': 'transparent'},
        showlegend=False
    )
    balances_forecast_lower = Scatter(
        name='', x=X_forecast, y=[last_balance] + list(forecast_lower),
        fill='tozeroy', fillcolor='#8881',
        line={'color': 'transparent'},
        showlegend=False
    )

    return [
        balances_forecast_lower,
        balances_forecast_upper,
        balances_forecast,
    ]




def top_html_graphs() -> Div:
    """
    Build weekly balance line graph and expense pie chart
    """
    monthly = transactions_monthly()
    expenses = [n for n in NAMES if monthly[n].sum() <= 0]
    pie = Pie(
        labels=expenses,
        values=[round(monthly[name].sum() * -1) for name in expenses],
        textinfo='label',
        marker={'colors': [color(name) for name in expenses]},
    )

    balances = TRANSACTIONS['amount']
    balances.index = TRANSACTIONS['date']
    balances = balances.groupby(pd.Grouper(freq='1w')).sum()
    # fill missing weeks with last balance
    balances = balances.asfreq('W').fillna(0).cumsum() + BALANCE

    balances_history = Scatter(
        name='balance',
        x=balances.index, y=balances,
        fill='tozeroy', fillcolor='#6662',
        line={'color': '#666'}, mode='lines',
    )

    balance_traces = [balances_history]
    balance_traces += forecast_traces(balances, weeks=52)

    layout = Layout(
        yaxis={'tickformat': ',.0f', 'rangemode': 'tozero'},
        xaxis={'tickformat': '%b %y'},
        title='Balance'
    )

    return Div([
        dcc.Graph(
            id='line',
            figure=Figure(data=balance_traces, layout=layout),
            style={'width': '66%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='pie',
            figure=Figure(data=[pie], layout=Layout(title='Expenses')),
            style={'width': '33%', 'display': 'inline-block'}
        ),
    ])



def main():
    app = dash.Dash()
    html_graphs = top_html_graphs()
    html_table = monthly_html_table()

    app.css.append_css({
        'external_url': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css'
    })
    app.layout = Div(children=[
        Div(H1(['Personal Finance ', html.Small('insights', style={'color':'#888'})]),
            className='page-header'),
        html_graphs,
        html_table
    ], style={'padding': '2%'})

    # firefox = Popen(['firefox', '--new-tab', 'http://127.0.0.1:8050'])
    app.run_server(debug=True)
    # firefox.terminate()


if __name__ == '__main__':
    main()
