from config import BALANCE, BALANCE_DATE, TRANSACTIONS, COLORS

from subprocess import Popen
import numpy as np
import pandas as pd
from typing import List

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components import Th, Thead, Td, Div, H1, Table, Tr, Span
import plotly
from plotly.graph_objs import Bar, Layout, Figure, Pie, Scatter


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
    expenses = [name for name in df.columns if df[name].sum() < 0]
    income = [name for name in df.columns if df[name].sum() > 0]
    df.index = [s.strftime('%b %y') for s in df.index]
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
        cells = [Th([circle(COLORS[cat]), cat], scope='row', style={'background': '#fafafa'})]
        for cell in row:
            style = {'color': COLORS['INC'] if cell >= 0 else COLORS['EXP'],
                     'text-align': 'right'}
            cells.append(Td(curr(cell or '') if cell != 0 else '', style=style))
        name_rows.append(Tr(cells))

    # growth row
    balances = list(df.sum().cumsum() + BALANCE)
    saving_rel = np.divide(balances, [BALANCE] + balances[:-1]) - 1
    saving_abs = np.array(balances) - np.array([BALANCE] + balances[:-1])

    growth_row = [Th('GROWTH', style={'background': '#f5f5f5', 'border-top': 'solid 2px #CCC'})]
    for sr, sa in zip(saving_rel, saving_abs):
        sa = Span(curr(sa), style={'color': COLORS['INC'] if sr >= 0 else COLORS['EXP'], 'font-weight': 'bold'})
        sr = Span(f'{perc(sr)}', style={'color': COLORS['INC'] + '88' if sr >= 0 else COLORS['EXP'] + '88'})
        sep = Span('  |  ', style={'color': '#CCC'})
        growth_row.append(Td([sr, sep, sa], style={'border-top': 'solid 2px #CCC', 'text-align': 'right'}))
    growth_row = Tr(growth_row)


    # total row
    total_row = [Th('TOTAL', style={'background': '#f5f5f5'})]
    for exp, inc in zip(df.loc[expenses].sum(), df.loc[income].sum()):
        sexp = Span(curr(exp), style={'color': COLORS['EXP'], 'font-weight': 'bold'})
        sinc = Span(curr(inc), style={'color': COLORS['INC'], 'font-weight': 'bold'})
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


def forecast_traces(balances, last_month, many_months=12) -> List[Scatter]:
    """
    Forecast next months based on balance history
    Exponential moving medians are used to forecast
    """
    last_balance = balances[-1]
    X_forecast = list(pd.date_range(last_month, periods=many_months, freq='MS'))

    growths_monthly = np.divide(balances, [BALANCE] + balances[:-1])
    # value recent ratios more with exponential moving median model
    growth_medians = pd.Series(growths_monthly).ewm(alpha=0.3).mean(how='median')
    low, median, high = growth_medians.quantile([0.25,0.5,0.75])

    def forecast(growth_factor):
        start = balances[-1]/growth_factor # first forecast should be just last balance
        return start * np.repeat(growth_factor, many_months).cumprod()

    # next 12 months forecast
    balances_forecast = Scatter(
        name='forecast', x=X_forecast, y=forecast(median),
        mode='lines',
        line={'dash': 'dash', 'color': COLORS['INC'] if median > 1 else COLORS['EXP']},
        opacity=0.8
    )
    balances_forecast_max = Scatter(
        name='', x=X_forecast, y=forecast(high if last_balance > 0 else low),
        fill='tonexty', fillcolor=(COLORS['INC'] if high > 1 else COLORS['EXP']) + '33',
        line={'color': 'transparent'},
        showlegend=False
    )
    steady_line = Scatter(
        name='', x=X_forecast, y=forecast(1),
        fill='tonexty', fillcolor=COLORS['EXP'] + '33' if 1 > low else '#0000',
        line={'color': 'transparent'},
        showlegend=False
    )
    balances_forecast_min = Scatter(
        name='', x=X_forecast, y=forecast(high if last_balance < 0 else low),
        fill='tozeroy', fillcolor='#8881' if 1 > low else '#0000',
        line={'color': 'transparent'},
        showlegend=False
    )

    return [
        balances_forecast_min,
        steady_line,
        balances_forecast_max,
        balances_forecast,
    ]




def top_html_graphs() -> Div:
    """
    Build balance line graph and expense pie chart
    """
    monthly = transactions_monthly()
    expenses = [name for name in monthly.columns if monthly[name].sum() < 0]
    pie = Pie(
        labels=monthly[expenses].columns,
        values=[round(monthly[name].sum() * -1) for name in expenses],
        textinfo='value+label',
        marker={'colors': [COLORS[name] for name in expenses]},
    )

    X_months_history = list(monthly.index)

    balances = list(monthly.sum(axis=1).cumsum() + BALANCE)
    balances_history = Scatter(
        name='balance',
        x=X_months_history, y=balances,
        fill='tozeroy', fillcolor='#6662',
        line={'color': '#666'}, mode='lines',
    )

    balance_traces = [balances_history]
    balance_traces += forecast_traces(balances, X_months_history[-1], 12)

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
