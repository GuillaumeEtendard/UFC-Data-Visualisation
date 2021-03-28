import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import base64
import pycountry


# Function to get the country code
def get_country_code(country):
    if 'South Korea' in country:
        return 'KOR'
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_3


app = dash.Dash(__name__)

# Data Preparation

df = pd.read_csv('ufc-master.csv', low_memory=False)
# Generate a datetime for the field date
df['date'] = pd.to_datetime(df['date'])
# Create a field year based on the datetime
df['year'] = df['date'].dt.year

df['country'] = df['country'].apply(lambda x: x.strip())
df.sort_values(by=['date'], ascending=False, inplace=True)

# Create the fights plot line
df_by_year = df.groupby(df.date.dt.year).agg('count')

fig_fights_year = px.line(df_by_year, x=df_by_year.index, y="R_fighter")
fig_fights_year.update_layout(xaxis_title="Year", yaxis_title="Number of Fights",
                              title='Total Number of Fights by Year')

# Create the bubble map plot
df_country = df.groupby(['country', 'year']).size().reset_index(name='total')
df_country.sort_values(by=['country', 'year'], inplace=True)

iso_map = {country: get_country_code(country) for country in df_country['country'].unique()}
df_country['country_code'] = df_country["country"].map(iso_map)

df_country = df_country.pivot_table(values='total', index='country_code', columns='year').stack(dropna=False).fillna(
    0).reset_index(name='total')
bubble_map = px.scatter_geo(df_country, locations="country_code", color="country_code",
                            hover_name="country_code", size="total",
                            animation_frame="year",
                            projection="natural earth",
                            )
bubble_map.update_layout(
    title_text="Fights location over the years",
    height=800
)

# Create the pie win rate
winner_values = df['Winner'].value_counts()
pie_winrate = px.pie(values=winner_values, names=['Red', 'Blue'], title="Red vs Blue Win rate",
                     color_discrete_sequence=px.colors.qualitative.Set1)

# Compute when the favorite is Red or Blue and when he wins or not
favWinsRed = df[(df['R_odds'] < 0) & (df['Winner'] == 'Red')].count()[0]
favWinsBlue = df[(df['B_odds'] < 0) & (df['Winner'] == 'Blue')].count()[0]
nonFavWinsRed = df[(df['R_odds'] > 0) & (df['Winner'] == 'Red')].count()[0]
nonFavWinsBlue = df[(df['B_odds'] > 0) & (df['Winner'] == 'Blue')].count()[0]
favWins = favWinsRed + favWinsBlue
nonFavWins = nonFavWinsRed + nonFavWinsBlue

# Make a pie with the favorite vs non favorite
pie_favorite = px.pie(values=[favWins, nonFavWins], names=['Favorite', 'Non Favorite'],
                      title="Favorite vs Non Favorite Win rate", color_discrete_sequence=['Green', 'Red'])

# Get the count of each finish details
df_finish = df
df_finish[['finish_details']] = df_finish['finish_details'].fillna('Other')
df_finish = df_finish.groupby(["finish", "finish_details"]).size().reset_index(name='total')


# Function to remove detail when the finish is a Decision (no more detail)
def fillUniqueNone(row):
    if 'DEC' in row.finish:
        row.finish_details = None
    return row


df_finish = df_finish.apply(lambda row: fillUniqueNone(row), axis=1)

# Create a sunburst with the fight finish details
sunburst_finish_details = px.sunburst(df_finish, path=['finish', 'finish_details'], values='total')
sunburst_finish_details.update_layout(height=600, title='Fights Finish Details')

df_country = df.groupby(['country']).size().reset_index(name='total')
bar_countries = px.bar(y=df_country.total, x=df_country.country)
bar_countries.update_layout(xaxis_title="Country", yaxis_title="Number of Fights",
                            title='Total Number of Fights by Country')

# Get all the fighters
all_fighters = pd.concat([df['R_fighter'], df['B_fighter']]).unique()

# Create a fighters dict
fighters_summary = dict()
for name in all_fighters:
    total_fights = df["Winner"][df["R_fighter"] == name].count() + df["Winner"][df["B_fighter"] == name].count()
    wins = df["Winner"][df["Winner"] == "Red"][df["R_fighter"] == name].count() + \
           df["Winner"][df["Winner"] == "Blue"][df["B_fighter"] == name].count()
    losses = total_fights - wins
    fighter = (df["R_fighter"] == name) | (df["B_fighter"] == name)
    fight_detail = df[fighter].iloc[0]
    if fight_detail.R_fighter == name:
        age = fight_detail.R_age
        rank = fight_detail['R_Pound-for-Pound_rank']
    else:
        age = fight_detail.B_age
        rank = fight_detail['B_Pound-for-Pound_rank']
    fighters_summary[name] = (
        total_fights, wins, losses, df[fighter].weight_class.unique()[0], age, rank, df[fighter].gender.unique()[0])

array_fighters = np.array(list(fighters_summary.values()))
# Create a dataframe with only the fighters
fighters_df = pd.DataFrame(
    {'name': fighters_summary.keys(), 'total_fights': array_fighters[:, 0], 'wins': array_fighters[:, 1],
     'losses': array_fighters[:, 2], 'weight': array_fighters[:, 3], 'age': array_fighters[:, 4],
     'rank': array_fighters[:, 5], 'gender': array_fighters[:, 6]})

fighters_df[["total_fights", 'wins', 'losses', 'age', 'rank']] = fighters_df[
    ["total_fights", 'wins', 'losses', 'age', 'rank']].apply(pd.to_numeric, errors='coerce')

# Keep only best fighters in one dataframe
biggest_fighters = fighters_df.sort_values(by=['total_fights'], ascending=False).head(5)

# Display best fighters in a stacked bar plot with the total fights, the wins and losses of each fighters
fig_biggest_fighters = px.bar(biggest_fighters, x="name", y=["wins", "losses"])
fig_biggest_fighters.update_layout(xaxis_title="Fighters", yaxis_title="Number of Fights with Wins and Losses",
                                   title='Biggest UFC Fighters',
                                   plot_bgcolor='#111111',
                                   paper_bgcolor='#111111',
                                   font_color='#7FDBFF',
                                   )

top_fighters = fighters_df[fighters_df['rank'] > 0].sort_values(by=['rank'])
top_fighters_male = top_fighters[top_fighters['gender'] == 'MALE']

# Create with a bar plot a Podium with the top 3 male fighters sorted by rank
fig_top_fighters_male = px.bar(
    x=[top_fighters_male.iloc[1]['name'], top_fighters_male.iloc[0]['name'], top_fighters_male.iloc[2]['name']],
    y=[2, 3, 1])
fig_top_fighters_male.update_layout(xaxis_title="Fighters", yaxis_title="Podium",
                                    title='Podium Pound for Pound (Male)',
                                    plot_bgcolor='#111111',
                                    paper_bgcolor='#111111',
                                    font_color='#7FDBFF',
                                    )

top_fighters_female = top_fighters[top_fighters['gender'] == 'FEMALE']
# Create with a bar plot a Podium with the top 3 female fighters sorted by rank
fig_top_fighters_female = px.bar(
    x=[top_fighters_female.iloc[1]['name'], top_fighters_female.iloc[0]['name'], top_fighters_female.iloc[2]['name']],
    y=[2, 3, 1])
fig_top_fighters_female.update_layout(xaxis_title="Fighters", yaxis_title="Podium",
                                      title='Podium Pound for Pound (Female)',
                                      plot_bgcolor='#111111',
                                      paper_bgcolor='#111111',
                                      font_color='#7FDBFF',
                                      )

fighters_by_weight = fighters_df.groupby(["weight"]).size().reset_index(name='total')

# Treemap with fights by weight class
treemap_fights = px.treemap(fighters_df, path=['weight'], values='total_fights')
treemap_fights.update_traces(texttemplate="%{label}<br>%{value} fights")
treemap_fights.update_layout(height=600, title='Treemap of Fights by Category')

# Treemap with fighters by weight class
treemap_fighters = px.treemap(fighters_by_weight, path=['weight'], values='total')
treemap_fighters.update_traces(texttemplate="%{label}<br>%{value} fighters")
treemap_fighters.update_layout(height=600, title='Treemap of Fighters by Category')

# Get only the title fights
df_title = df[df.title_bout]
df_title.dropna(inplace=True, subset=['finish_round'])

fig_time_series = go.Figure()

fig_time_series.add_trace(
    go.Scatter(x=list(df_title.date), y=list(df_title.finish_round),
               customdata=df['R_fighter'] + ' vs ' + df['B_fighter'],
               hovertemplate="Fight:%{customdata}<br>Date: %{x}<br>Number of rounds: %{y}",
               ))

# Add range slider
fig_time_series.update_layout(
    title_text="Number of rounds in Fights for title",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ),
    plot_bgcolor='#111111',
    paper_bgcolor='#111111',
    font_color='#7FDBFF',
    height=600
)
# Read UFC Logo
image_filename = 'assets/img/UFC_Logo.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# Tab css
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'background': '#000',
    'padding': '10px',
    'color': 'white',
    'fontSize': '25px'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#d20909',
    'color': 'white',
    'padding': '10px',
    'fontSize': '25px',
    'fontWeight': 'bold',
}

# App layout
app.layout = html.Div(
    style={'backgroundColor': '#f9fbfd'},
    children=[
        html.Div(
            style={'textAlign': 'center', 'marginBottom': 20},
            children=[
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    width=200,
                )
            ]
        ),
        html.H3(
            'A Dashboard about UFC Fights & UFC Fighters, the most famous MMA Organisation.',
            style={'textAlign': 'center', 'marginBottom': 10, 'font-family': 'Garamond'}
        ),
        html.H4(
            'Evolve throughout the UFC Event : Early Prelims, Prelims & Main Card. And don\'t miss the Main Event !',
            style={'textAlign': 'center', 'marginBottom': 20, 'font-family': 'Garamond'}
        ),
        html.Div([
            dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
                dcc.Tab(label='Early Prelims', value='tab-1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Prelims', value='tab-2', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Main Card', value='tab-3', style=tab_style, selected_style=tab_selected_style),
            ], className='tabs_styles'),
            html.Div(id='tabs-content-props')
        ])
    ]
)


# App callback for the scatter plot with range slider
@app.callback(
    Output("scatter-plot", "figure"),
    [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    low, high = slider_range
    # Get the top 100 fighters
    fighters_df_limit = fighters_df.sort_values(by=['total_fights'], ascending=False).head(100)
    # Mask with the slider range
    mask = (fighters_df_limit['age'] > low) & (fighters_df_limit['age'] < high)
    # Create the fighters scatter plot
    fig = px.scatter(
        fighters_df_limit[mask], x="age", y="total_fights",
        color="name", size='total_fights',
        hover_data=['age'])
    fig.update_layout(title='Fighters Age comparison', xaxis_title='Age', yaxis_title='Total Fights', height=800)
    return fig


# App callback for the dropdown of fighters
@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('fighters-dropdown', 'value')])
# Function for updating the dropdown results
def update_output(value):
    # Get the fighter selected
    df_fighter = df[(df['R_fighter'] == value) | (df['B_fighter'] == value)]

    # Function to know if the fighter is the winner of each fight
    def is_winner(row):
        if row.R_fighter == value:
            row['isWinner'] = row.Winner == 'Red'
        else:
            row['isWinner'] = row.Winner == 'Blue'
        return row

    df_fighter = df_fighter.apply(lambda row: is_winner(row), axis=1).sort_values(by=['date']).reset_index()

    # Create a line plot with the number of fights over the years
    fig_fighter = px.line(df_fighter, x=df_fighter.date, y=df_fighter.index)
    fig_fighter.update_layout(xaxis_title="Date", yaxis_title="Number of Fights",
                              title='Total Fights by Year')

    # Create a scatter plot with the results of the fights
    bar_fighter = go.Figure(data=go.Scatter(
        x=df_fighter.date,
        y=df_fighter.isWinner,
        mode='markers',
        marker=dict(
            color='LightSkyBlue',
            size=20
        ),
    ))
    bar_fighter.update_yaxes(categoryorder='category ascending')
    bar_fighter.update_layout(xaxis_title="Date", yaxis_title="Did he win (False/True)",
                              title='Is he the winner of the fights ?')

    # Dict of fighters photos
    images = {
        'Conor McGregor': 'https://wallpaper.dog/large/326623.jpg',
        'Khabib Nurmagomedov': 'https://wallpaperaccess.com/full/2053184.jpg',
        'Jon Jones': 'https://pbs.twimg.com/media/EQxFWrsWkAcOCEg.jpg',
        'Georges St-Pierre': 'https://i.pinimg.com/originals/a2/9e/50/a29e50ffbf478ec2ec3fb1e15890c329.jpg',
        'Demetrious Johnson': 'https://i.pinimg.com/originals/b9/43/e9/b943e978f9070afcc6f496e482827da4.jpg',
        'Anderson Silva': 'https://i.pinimg.com/originals/df/71/78/df7178e3ae87f6f7a6832619b302b64f.jpg'
    }

    # Display the generated graphs
    return html.Div([
        html.Div([
            html.Img(
                src=images[value],
                className='img'
            )],
            className='col-4',
        ),
        html.Div([
            html.Div([
                html.Div(
                    dcc.Graph(
                        figure=fig_fighter,
                        className='graph_dark'
                    ),
                    className='graph_container'
                )],
                className='col-12',
            ),
            html.Div([
                html.Div(
                    dcc.Graph(
                        figure=bar_fighter,
                        className='graph_dark'
                    ),
                    className='graph_container'
                )],
                className='col-12',
            ),
        ], className='col-8'),
    ], className='row')


# App callback for the tabs
@app.callback(Output('tabs-content-props', 'children'),
              Input('tabs-styled-with-props', 'value'))
# Render the content of each tab
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=fig_fights_year,
                            className='graph_style'
                        ),
                        className='graph_container'
                    )],
                    className='col-6',
                ),
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=bar_countries,
                            className='graph_style'
                        ),
                        className='graph_container'
                    )],
                    className='col-6',
                )],
                className='row'
            ),
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=bubble_map,
                            className='graph_style'
                        ),
                        className='graph_container'
                    )],
                    className='col-12',
                )],
                className='row'
            ),
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=pie_winrate,
                            className='graph_style'
                        ),
                        className='graph_container'
                    )],
                    className='col-6',
                ),
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=pie_favorite,
                            className='graph_style'
                        ),
                        className='graph_container'
                    )],
                    className='col-6',
                )],
                className='row'
            )
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=treemap_fights,
                            className='graph_style'
                        ), className='graph_container')
                ], className='col-6'),
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=treemap_fighters,
                            className='graph_style'
                        ), className='graph_container')
                ], className='col-6')
            ], className='row'),
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=sunburst_finish_details,
                            className='graph_style'
                        ), className='graph_container')
                ], className='col-12')
            ], className='row'),
            html.Div([
                dcc.Graph(id="scatter-plot"),
                html.P("Age:"),
                dcc.RangeSlider(
                    id='range-slider',
                    min=20, max=40, step=1,
                    marks={20: '20', 40: '40'},
                    value=[20, 40]
                ),
            ]),
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=fig_top_fighters_male,
                            className='graph_dark'
                        ),
                        className='graph_container'
                    )],
                    className='col-4',
                ),
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=fig_top_fighters_female,
                            className='graph_dark'
                        ),
                        className='graph_container'
                    )],
                    className='col-4',
                ),
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=fig_biggest_fighters,
                            className='graph_dark'
                        ),
                        className='graph_container'
                    )],
                    className='col-4',
                ),
            ], className='row'),
            html.Div([
                html.Div([
                    html.Div(
                        dcc.Graph(
                            figure=fig_time_series,
                            className='graph_dark'
                        ),
                        className='graph_container'
                    )],
                    className='col-12',
                ),
            ], className='row'),
            html.H1('Main Event', className='title'),
            html.H3('Select your favorite Fighter among the most famous', className='text-center'),
            dcc.Dropdown(
                id='fighters-dropdown',
                options=[
                    {'label': 'Conor McGregor', 'value': 'Conor McGregor'},
                    {'label': 'Jon Jones', 'value': 'Jon Jones'},
                    {'label': 'Anderson Silva', 'value': 'Anderson Silva'},
                    {'label': 'Khabib Nurmagomedov', 'value': 'Khabib Nurmagomedov'},
                    {'label': 'Demetrious Johnson', 'value': 'Demetrious Johnson'},
                    {'label': 'Georges St-Pierre', 'value': 'Georges St-Pierre'}
                ],
                value='Conor McGregor'
            ),
            html.Div(id='dd-output-container'),

        ], className='dark')


if __name__ == '__main__':
    app.run_server()
