# Import external libraries
import sqlite3
from collections import Counter
from collections import defaultdict
from itertools import combinations
import pandas as pd # type: ignore
from datetime import date
from dateutil.relativedelta import relativedelta # type: ignore
from lxml import etree # type: ignore
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update # type: ignore
import dash_bootstrap_components as dbc # type: ignore
import dash_auth # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
from plotly_calplot import calplot # type: ignore
import sqlparse # type: ignore
from sqlparse.tokens import Keyword  # type: ignore
import math
from user_config import VALID_USERNAME_PASSWORD_PAIRS
from io import StringIO # type: ignore
import gzip
import base64
import os
from sqlalchemy import create_engine # type: ignore

#---------------------------------------------------------------------
# Global settings
#---------------------------------------------------------------------

LOCAL_DB_DUMP = True
LOCAL_ONLY = True
color_palette = px.colors.qualitative.Plotly
styles = {
    'header': {'className': 'py-3'},
    'container': {'className': 'my-4'},
    'pagination': {'style': {'display': 'flex', 'justifyContent': 'flex-end', 'paddingRight': '20px', 'paddingBottom': '20px'}}
}

#---------------------------------------------------------------------
# Load data
#---------------------------------------------------------------------

if (LOCAL_DB_DUMP):
    # Load database from local database dump file
    con = sqlite3.connect("dbdump.db")
    cur = con.cursor()
    # Use a SQL query to load data into a Pandas DataFrame
    query = "SELECT * FROM qt_query_master;"
    df_qt_query_master = pd.read_sql_query(query, con)
else:
    # Load database from associated i2b2 database container
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PW")
    PORT_DB = "5432"
    db_user = 'postgres'
    db_host = "i2b2-{}-db".format(os.getenv("INSTANCE_NUM"))
    db_database = 'i2b2'
    connection_url = f"postgresql://{db_user}:{POSTGRES_PASSWORD}@{db_host}:{PORT_DB}/{db_database}"
    engine = create_engine(connection_url)
    # Use a SQL query to load data into a Pandas DataFrame
    query = "SELECT * FROM i2b2demodata.qt_query_master;"
    df_qt_query_master = pd.read_sql_query(query, engine)

#---------------------------------------------------------------------
# Define time range slider
#---------------------------------------------------------------------

# Convert 'create_date'  to datetime format
df_qt_query_master['create_date'] = pd.to_datetime(df_qt_query_master['create_date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
# Define range by earliest and latest create_date
range_start_date = df_qt_query_master['create_date'].min()
range_end_date = df_qt_query_master['create_date'].max()

# Calculate monthly marks
def calculate_monthly_marks(range_start_date, range_end_date):
    # Helper function to add months
    def add_months(sourcedate, months):
        return sourcedate + relativedelta(months=months)
    
    # Calculate number of months between start and end of the range (add 1, so queries in the last months are also available)
    num_months_in_range = (range_end_date.year - range_start_date.year) * 12 + range_end_date.month - range_start_date.month + 1

    # Define interval between labels
    def determine_label_interval(n):
        if n < 24: # For less than 2 years, set labels every month
            return 1 
        elif n < 36: # For less than 3 years, set labels every 2 months
            return 2
        elif n < 60: # For less than 5 years, set labels every 3 months
            return 3
        elif n < 120: # For less than 10 years, set labels every 6 months
            return 6
        elif n < 240: # For less than 20 years, set labels every 12 months
            return 12
        else: # For more than 20 years, set labels every 24 months
            return 24
    label_interval = determine_label_interval(num_months_in_range)

    # Generate marks and their labels, and store associated dates
    marks = {} # Generate marks for the slider 
    date_values = {} # Store specific dates in addition to creating marks (as some have empty labels)
    for i in range(num_months_in_range + 1):
        month_date = add_months(range_start_date, i)
        date_values[i] = month_date.strftime('%Y-%m')
        # Only add label every `label_interval` months
        if i % label_interval == 0:
            # Add mark with a label
            marks[i] = {'label': month_date.strftime('%Y-%m')} 
        else:
            # Add mark without a label
            marks[i] = {'label': ''}
    
    # Return marks (i.e. markings on the range slider, with and without labels), date_values (associated dates for each mark), and num_months_in_range
    return marks, date_values, num_months_in_range

# Get marks, date_values and maximum range for the slider
marks, date_values, max_range = calculate_monthly_marks(range_start_date, range_end_date)

# RangeSlider configuration
date_range_slider = dcc.RangeSlider(
    id = 'date-range-slider', # Note: id needed, as value needs to be accessed as Input by update_analytics_time_range
    min=0,
    max=max_range,
    step=1,
    marks=marks,
    value=[0, max_range],
    pushable=1
)

#---------------------------------------------------------------------
# Helper functions
#---------------------------------------------------------------------

# Encode and compress data
def compress_and_encode_dataframe(df):
    json_str = df.to_json(orient='split')
    json_bytes = json_str.encode('utf-8')
    compressed = gzip.compress(json_bytes)
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded

# Decode and decompress data
def decode_and_decompress_dataframe(encoded_compressed_data):
    if encoded_compressed_data is None:
        print("No data available")
    compressed = base64.b64decode(encoded_compressed_data)
    decompressed = gzip.decompress(compressed)
    json_str = decompressed.decode('utf-8')
    df = pd.read_json(StringIO(json_str), orient='split')
    return df

# Parse XML and extract 'item_key' values
def extract_item_keys(xml_string):
    try:
        xml_bytes = xml_string.encode('utf-8')
        root = etree.fromstring(xml_bytes)
        item_keys = root.xpath('//item/item_key/text()')
        return item_keys
    except Exception as e:
        print(f"Error processing XML: {e}")
        return []

# Parse XML and count existing query constraints(e.g. date constraints, value constraints, total item occurance constraints)
def count_query_constraint(xml_string):
    try:
        xml_bytes = xml_string.encode('utf-8')
        root = etree.fromstring(xml_bytes)
        date_constraint = len(root.xpath('//item/constrain_by_date')) # get all date constraints
        value_constraint = len(root.xpath('//item/constrain_by_value')) # get all value constraints
        total_item_occurrences = root.xpath('//panel/total_item_occurrences/text()') # get item occurance setting for each panel
        count_item_occurances_constraints = sum(int(n) > 1 for n in total_item_occurrences) # usally 1, if >1 count as contraint was added
        sum_of_constraints = date_constraint+value_constraint+count_item_occurances_constraints
        return sum_of_constraints
    except Exception as e:
        print(f"Error processing XML: {e}")
        return []
    
# Parse XML and check if a temporal query was performed by returning the number of temporal constraints (based on existence of a subquery contraint, that defines a temporal relationship between two events)
def count_temporal_query_constraints(xml_string):
    try:
        xml_bytes = xml_string.encode('utf-8')
        root = etree.fromstring(xml_bytes)
        subquery_constraint = len(root.xpath('//subquery_constraint')) # get all subquery constraints
        return subquery_constraint
    except Exception as e:
        print(f"Error processing XML: {e}")
        return []

# Create placeholder figure if no or not enough data  
def no_data_placeholder(title_text, message_text):
    fig = go.Figure()
    fig.add_annotation(
        text=message_text,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        title_text=title_text,
        height= 100,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    # Create dash graph element
    figure = dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=fig
            ),
            md=12
        )
    ])
    return figure

#---------------------------------------------------------------------
# Plot in Dash app
#---------------------------------------------------------------------

if LOCAL_ONLY:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
else:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], requests_pathname_prefix="/usage/", suppress_callback_exceptions=True)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
app.layout = dbc.Container([
    dbc.Container([
        html.H1("i2b2 Usage Dimensions", **styles['header']),
        html.H5(id='selected-period-display', className='pb-3'),
        dcc.Store(id='df-qt-query-master-filtered')
    ]),
    date_range_slider,
    dcc.Loading(
        id="loading-overlay",
        fullscreen=True,
        overlay_style={'opacity': '0.8'},
        children=[
            dbc.Container([
                html.H4("Queries and Users", **styles['header']),
                dbc.Container(id='overall-numbers', className='my-3'),
                html.Div(
                    dbc.Pagination(id='pagination-users', max_value=5, first_last=True, previous_next=True, fully_expanded=False, size="sm"),
                    **styles['pagination']
                ),
                dbc.Container(id='calplot-figure', **styles['container']),
                dbc.Container(id='bar-charts-queries-and-users', **styles['container']),
                dbc.Container(id='stacked-area-chart', **styles['container'])
            ]),
            dbc.Container([
                html.H4("Concepts", **styles['header']),
                dbc.Container(id='bar-chart-concepts', className='my-3'),
                html.Div(
                    dbc.Pagination(id='pagination-concepts', max_value=5, first_last=True, previous_next=True, fully_expanded=False, size="sm"),
                    **styles['pagination']
                ),
                dbc.Container(id='treemap-graph', **styles['container']),
                dbc.Container(id='table-concepts', **styles['container']),
                html.Div(
                    dbc.Pagination(id='pagination-concept-table', max_value=5, first_last=True, previous_next=True, fully_expanded=False, size="sm"),
                    **styles['pagination']
                )
            ]),
            dbc.Container([
                html.H4("Complexity", **styles['header']),
                dbc.Container(id='complexity', className='my-3'),
                dbc.Container(id='user-complexity', **styles['container']),
                html.Div(
                    dbc.Pagination(id='pagination-complexity', max_value=5, first_last=True, previous_next=True, fully_expanded=False, size="sm"),
                    **styles['pagination']
                )
            ])
        ]
    ),
], fluid=True)

#---------------------------------------------------------------------
# Update time range for analytics
#---------------------------------------------------------------------
@app.callback(
    [
        Output('selected-period-display', 'children'),
        Output('df-qt-query-master-filtered', 'data')
    ],
    [
        Input('date-range-slider', 'value')
    ]
)
def update_analytics_time_range(slider_range):

    # Convert slider marks back to datetime
    selected_start_date = pd.to_datetime(date_values[slider_range[0]])
    selected_end_date = pd.to_datetime(date_values[slider_range[1]])

    # Display selected time period as text
    seleceted_period_display = f"Analytics for a time period between {selected_start_date.strftime('%Y-%m-%d')} and {selected_end_date.strftime('%Y-%m-%d')}"

    # Filter the DataFrame
    df_qt_query_master_filtered = df_qt_query_master[
        (pd.to_datetime(df_qt_query_master['create_date']) >= selected_start_date) &
        (pd.to_datetime(df_qt_query_master['create_date']) <= selected_end_date)
    ].copy()

    # Compress dataframe
    filtered_df = compress_and_encode_dataframe(df_qt_query_master_filtered)

    return seleceted_period_display, filtered_df

#---------------------------------------------------------------------
# 1. Update query and user analytics
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# 1.1.Show overall numbers of users, queries and top users
#---------------------------------------------------------------------
@app.callback(
    [
        Output('overall-numbers', 'children'),
        Output('pagination-users', 'max_value'),
        Output('pagination-users', 'active_page')
    ],
    [
        Input('df-qt-query-master-filtered', 'data'),
        Input('pagination-users', 'active_page')
    ],
    [
        State('pagination-users', 'active_page')
    ]
)
def update_overall_numbers(df_qt_query_master_filtered, active_page, current_page):
    
    # Pagination functionalities
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'df-qt-query-master-filtered':
        active_page = 1
    else:
        active_page = current_page if current_page else 1
    
    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)

    # Convert 'create_date'  to datetime format
    df_qt_query_master_filtered['create_date'] = pd.to_datetime(df_qt_query_master_filtered['create_date'] , unit='ms')

    # Calculate the number of queries in the filtered DataFrame
    count_queries = df_qt_query_master_filtered['query_master_id'].nunique()

    # Calculate the number of unique users in the filtered DataFrame
    num_unique_users = df_qt_query_master_filtered['user_id'].nunique()

    # Number of queries by user, sorted in descending order
    top_users = df_qt_query_master_filtered.groupby('user_id').size().sort_values(ascending=False)

    # Create a DataFrame for the overall numbers table
    overall_numbers_data = {
        'Metric': ['Number of queries', 'Number of unique users'],
        'Value': [count_queries, num_unique_users]
    }
    df_overall_numbers = pd.DataFrame(overall_numbers_data)

    # Create a DataFrame for the top users table
    # Calculate total pages
    page_size = 5
    total_pages = (len(top_users) + page_size - 1) // page_size

    # Handle pagination
    start_row = (active_page - 1) * page_size
    end_row = start_row + page_size
    top_users_data = {
        'User': [f"User {user_id}" for user_id in top_users.index[start_row:end_row]],
        'Number of Queries': top_users.values[start_row:end_row]
    }
    df_top_users = pd.DataFrame(top_users_data)

    # Create dbc tables
    overall_numbers = dbc.Container([
        dbc.Row([
            dbc.Col(
                dbc.Table(
                    children=[
                        html.Thead(html.Tr([html.Th(col) for col in df_overall_numbers.columns])),
                        html.Tbody([
                            html.Tr([
                                html.Td(row[col]) for col in df_overall_numbers.columns
                            ]) for row in df_overall_numbers.to_dict('records')
                        ])
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    style={
                        'width': '50%',
                        'fontSize': '12px'
                    }
                ),
                md=12
            )
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Table(
                    children=[
                        html.Thead(html.Tr([html.Th(col) for col in df_top_users.columns])),
                        html.Tbody([
                            html.Tr([
                                html.Td(row[col]) for col in df_top_users.columns
                            ]) for row in df_top_users.to_dict('records')
                        ])
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    style={
                        'fontSize': '12px'
                    }
                ),
                md=12
            )
        ])
    ])
    return overall_numbers, total_pages, active_page

#---------------------------------------------------------------------
# 1.2. Calender Plot
#---------------------------------------------------------------------
@app.callback(
    [
        Output('calplot-figure', 'children')
    ],
    [
        Input('df-qt-query-master-filtered', 'data')
    ]
)
def update_calplot(df_qt_query_master_filtered):
    
    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)

    # Convert 'create_date'  to datetime format
    df_qt_query_master_filtered['create_date'] = pd.to_datetime(df_qt_query_master_filtered['create_date'] , unit='ms')
    
    # Create calplot
    if not df_qt_query_master_filtered.empty:
        # Create new column date
        df_qt_query_master_dates= df_qt_query_master_filtered.copy()
        df_qt_query_master_dates['date'] = df_qt_query_master_dates['create_date'].dt.date
        
        # Group by date and count queries
        daily_queries = df_qt_query_master_dates.groupby('date').size().reset_index(name='count')
        
        # Ensure the date column is datetime format
        daily_queries['date'] = pd.to_datetime(daily_queries['date'])

        # Calendar heatmap visualization
        fig = calplot(
            daily_queries,
            x='date',
            y='count',
            colorscale='dense',
            gap=0.5,
            years_title=True,
            month_lines_width=0.5 
        )

        # Tooltip
        fig.update_traces(hovertemplate='Week: %{x}<br>#Queries: %{z}')
    
        calplot_figure = dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig
                ),
                md=12
            )
        ])
    else:
        # No data placeholder figure
        calplot_figure=no_data_placeholder("Calender Plot", "No queries in the indicated time period")
    
    return [calplot_figure]

#---------------------------------------------------------------------
# 1.3. Bar charts queries and users by week
#---------------------------------------------------------------------
@app.callback(
    [
        Output('bar-charts-queries-and-users', 'children')
    ],
    [
        Input('df-qt-query-master-filtered', 'data')
    ]
)
def update_bar_charts_queries_and_users(df_qt_query_master_filtered):
    
    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)
    # Convert 'create_date'  to datetime format
    df_qt_query_master_filtered['create_date'] = pd.to_datetime(df_qt_query_master_filtered['create_date'] , unit='ms')

    # Create bar charts
    if not df_qt_query_master_filtered.empty:
        # Create new column with weeks
        df_qt_query_master_weeks= df_qt_query_master_filtered.copy()
        df_qt_query_master_weeks['week'] = df_qt_query_master_weeks['create_date'].dt.to_period('W').apply(lambda r: r.start_time)

        # Group by week and user_id, then count queries
        grouped_data = df_qt_query_master_weeks.groupby(['week', 'user_id']).size().reset_index(name='query_count')

        # Sum the query count per week to get the total number of queries per week
        total_queries_per_week = grouped_data.groupby('week')['query_count'].sum().reset_index()

        # Count unique users per week
        unique_users_per_week = grouped_data.groupby('week')['user_id'].nunique().reset_index()
        unique_users_per_week.columns = ['week', 'unique_users']

        # Create figure with two subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Number of Queries per Week", "Number of Unique Users per Week"))
        fig.add_trace(
            go.Bar(x=total_queries_per_week['week'], y=total_queries_per_week['query_count'], name='Queries', marker_color=color_palette[0]),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=unique_users_per_week['week'], y=unique_users_per_week['unique_users'], name='Users', marker_color=color_palette[1]),
            row=1, col=2
        )
        fig.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 1209600000], value="W%W %Y"), # split at two weeks
                dict(dtickrange=[1209600000, None], value="%b %Y")
            ],
            row=1, col=1
        )
        fig.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 1209600000], value="W%W %Y"), # split at two weeks
                dict(dtickrange=[1209600000, None], value="%b %Y")
            ],
            row=1, col=2
        )
        fig.update_layout(
            height=400
        )

        # Create dash graph element
        bar_charts_queries_and_users = dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig
                ),
                md=12
            )
        ])
    else:
        # No data placeholder figure
        bar_charts_queries_and_users = no_data_placeholder("Bar charts queries and users", "No queries in the indicated time period")
    
    return [bar_charts_queries_and_users]

#---------------------------------------------------------------------
# 1.4. Stacked area chart
#---------------------------------------------------------------------
@app.callback(
    [
        Output('stacked-area-chart', 'children')
    ],
    [
        Input('df-qt-query-master-filtered', 'data')
    ]
)
def update_stacked_area_chart(df_qt_query_master_filtered):
    
    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)
    # Convert 'create_date'  to datetime format
    df_qt_query_master_filtered['create_date'] = pd.to_datetime(df_qt_query_master_filtered['create_date'] , unit='ms')

    # Create new column date
    df_qt_query_master_dates=df_qt_query_master_filtered.copy()
    df_qt_query_master_dates['date'] = df_qt_query_master_dates['create_date'].dt.date

    # Check if any user_id has at least two different dates
    if df_qt_query_master_dates.groupby('user_id')['date'].nunique().max() > 1:
        
        # Group by dates
        grouped_data = df_qt_query_master_dates.groupby(['date', 'user_id']).size().reset_index(name='query_count')

        # Pivot the data for stacked area chart
        pivot_data = grouped_data.pivot(index='date', columns='user_id', values='query_count').fillna(0)

        # Plot stacked area chart
        fig = px.area(pivot_data, facet_col_wrap=4, color_discrete_sequence=color_palette)
        fig.update_layout(
            title="Number of queries per user over time",
            xaxis_title="Time",
            yaxis_title="Number of Queries",
            legend_title="User ID",
            showlegend=True,
            height=400
        )

        # Create dash graph element
        stacked_area_chart = dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig
                ),
                md=12
            )
        ])
    else:
        # No data placeholder figure
        stacked_area_chart = no_data_placeholder("Number of queries per user over time", "No or not enough queries in the indicated time period")

    return [stacked_area_chart]

#---------------------------------------------------------------------
# 2. Update frequently queried concept analytics
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# 2.1. Frequently queried concepts
#---------------------------------------------------------------------
@app.callback(
    [
        Output('bar-chart-concepts', 'children'),
        Output('pagination-concepts', 'max_value'),
        Output('pagination-concepts', 'active_page')
    ],
    [
        Input('df-qt-query-master-filtered', 'data'),
        Input('pagination-concepts', 'active_page')
    ],
    [
        State('pagination-concepts', 'active_page')
    ]
)
def update_bar_charts_concepts(df_qt_query_master_filtered, active_page, current_page):

    # Pagination functionalities
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'df-qt-query-master-filtered':
        active_page = 1
    else:
        active_page = current_page if current_page else 1
    
    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)
    
    # Create barchart
    if not df_qt_query_master_filtered.empty:
        # Apply the function to each row in the 'request_xml' column to get item keys (i.e. specific concept paths)
        df_qt_query_master_item_keys= df_qt_query_master_filtered.copy()
        df_qt_query_master_item_keys['item_keys'] = df_qt_query_master_item_keys['request_xml'].apply(extract_item_keys)

        # Flatten the list of item_keys to count occurrences
        item_key_counts = [key for sublist in df_qt_query_master_item_keys['item_keys'] for key in sublist]

        # Create a counter dictionary from the list
        item_key_counter = Counter(item_key_counts)

        # Create a DataFrame from the counter
        df_item_keys = pd.DataFrame(list(item_key_counter.items()), columns=['Item_Key', 'Count'])

        # Extract last two elements of the item key path
        df_item_keys['Label'] = df_item_keys['Item_Key'].apply(lambda x: '\\'.join(x.split('\\')[-3:]))

        # Sort by count in descending order
        df_item_keys_sorted = df_item_keys.sort_values(by='Count', ascending=False)

        # Calculate total pages
        page_size = 15
        total_pages = (len(df_item_keys_sorted) + page_size - 1) // page_size

        # Handle pagination
        start_row = (active_page - 1) * page_size
        end_row = start_row + page_size

        # Get the DataFrame slice for the current page
        df_page = df_item_keys_sorted.iloc[start_row:end_row]

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_page['Count'],
                y=df_page['Label'],
                text=df_page['Count'],
                hovertext=df_page['Item_Key'],
                orientation='h',
                name='Concept Counts',
                marker_color=color_palette[0],
                width=0.8
            )
        )
        
        fig.update_traces(
            hoverinfo="text"
        )
        
        fig.update_layout(
            height=(len(df_page) * 30) + 180,  #Hack to show bar traces with similar width
            title=f'Most Frequently Queried Concepts',
            showlegend=False,
            xaxis_title='Count',
            yaxis_title='Concept',
            yaxis=dict(
                categoryorder='total ascending',  
                automargin=True
            )
        )

        # Create dash graph element
        bar_charts_concepts =  dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig
                ),
                md=12
            )
        ])
    else:
        # No data placeholder figure
        bar_charts_concepts = no_data_placeholder("Most Frequently Queried Concepts", "No queries in the indicated time period")
        total_pages=0
    
    return bar_charts_concepts, total_pages, active_page

#---------------------------------------------------------------------
# 2.2 Tree Map
#---------------------------------------------------------------------
@app.callback(
    [
        Output('treemap-graph', 'children')
    ],
    [
        Input('df-qt-query-master-filtered', 'data')
    ]
)
def update_treemap(df_qt_query_master_filtered):

    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)

    # Create treemap
    if not df_qt_query_master_filtered.empty:
        # Apply the function to each row in the 'request_xml' column to get item keys (i.e. specific concept paths)
        df_qt_query_master_item_keys= df_qt_query_master_filtered.copy()
        df_qt_query_master_item_keys['item_keys'] = df_qt_query_master_item_keys['request_xml'].apply(extract_item_keys)
        
        # Flatten the list of item_keys to count occurrences
        item_key_counts = [key for sublist in df_qt_query_master_item_keys['item_keys'] for key in sublist]
        item_key_counter = Counter(item_key_counts)

        data_for_treemap = {
            'ids': ['All Concepts'],
            'labels': ['All Concepts'],
            'parents': [''],
            'values': [0]
        }

        cumulative_values = defaultdict(int)

        # Process each unique path for ids, labels, parents, and values
        for item_key, count in item_key_counter.items():
            parts = [part for part in item_key.split('\\') if part]
            for i in range(len(parts)):
                current_path = '\\'.join(parts[:i+1])
                parent_path = '\\'.join(parts[:i]) if i > 0 else 'All Concepts'
                
                cumulative_values[current_path] += count

                if current_path not in data_for_treemap['ids']:
                    data_for_treemap['ids'].append(current_path)
                    data_for_treemap['labels'].append(parts[i])
                    data_for_treemap['parents'].append(parent_path)
                    data_for_treemap['values'].append(0)

        # Assign cumulative values
        for idx, id_ in enumerate(data_for_treemap['ids']):
            data_for_treemap['values'][idx] = cumulative_values[id_]

        # Update the root value
        data_for_treemap['values'][0] = sum(cumulative_values[key] for key in cumulative_values if '\\' not in key)

        # Create DataFrame from the processed data
        df_treemap = pd.DataFrame(data_for_treemap).iloc[1:]
        
        fig =  go.Figure()
        fig.add_trace(go.Treemap(
            ids = df_treemap.ids,
            labels = df_treemap.labels,
            parents = df_treemap.parents,
            maxdepth=3,
            root_color="lightgrey"
        ))
        fig.update_layout(height=800)

        # Create dash graph element
        treemap = dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig
                ),
                md=12
            )
        ])
    else:
        # No data placeholder figure
        treemap = no_data_placeholder("Treemap", "No queries in the indicated time period")
    
    return [treemap]

#---------------------------------------------------------------------
# 2.3. Co-occurances of frequently queried concepts
#---------------------------------------------------------------------
@app.callback(
    [
        Output('table-concepts', 'children'),
        Output('pagination-concept-table', 'max_value'),
        Output('pagination-concept-table', 'active_page')
    ],
    [
        Input('df-qt-query-master-filtered', 'data'),
        Input('pagination-concept-table', 'active_page')
    ],
    [
        State('pagination-concept-table', 'active_page')
    ]
)
def update_table_concepts(df_qt_query_master_filtered, active_page, current_page):

    # Pagination functionalities
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'df-qt-query-master-filtered':
        active_page = 1
    else:
        active_page = current_page if current_page else 1

    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)

    # Get frequently queried concepts (same as in 2.1)
    df_qt_query_master_item_keys= df_qt_query_master_filtered.copy()
    df_qt_query_master_item_keys['item_keys'] = df_qt_query_master_item_keys['request_xml'].apply(extract_item_keys)
    item_key_counts = [key for sublist in df_qt_query_master_item_keys['item_keys'] for key in sublist]
    item_key_counter = Counter(item_key_counts)
    df_item_keys = pd.DataFrame(list(item_key_counter.items()), columns=['Item_Key', 'Count'])
    df_item_keys['Label'] = df_item_keys['Item_Key'].apply(lambda x: '\\'.join(x.split('\\')[-3:]))
    df_item_keys_sorted = df_item_keys.sort_values(by='Count', ascending=False)

    # Create a list of item_key sets
    item_key_sets = df_qt_query_master_item_keys['item_keys'].tolist()

    # Function to count co-occurrences
    def count_co_occurrences(item_key_sets):
        co_occurrence_count = Counter()
        for item_set in item_key_sets:
            for (item1, item2) in combinations(sorted(set(item_set)), 2):
                co_occurrence_count[(item1, item2)] += 1
        return co_occurrence_count

    # Count co-occurrences
    co_occurrences = count_co_occurrences(item_key_sets)

    if not df_item_keys_sorted.empty and co_occurrences:
        # Convert to dataframe
        df_co_occurrences = pd.DataFrame(((k[0], k[1], v) for k, v in co_occurrences.items()), columns=['Item1', 'Item2', 'Count'])

        # Adding overall occurrences to the co-occurrences DataFrame for reference
        df_co_occurrences['Overall_Count1'] = df_co_occurrences['Item1'].map(dict(zip(df_item_keys_sorted['Item_Key'], df_item_keys_sorted['Count'])))
        df_co_occurrences['Overall_Count2'] = df_co_occurrences['Item2'].map(dict(zip(df_item_keys_sorted['Item_Key'], df_item_keys_sorted['Count'])))

        def top_co_occurrences(row, df, max_items=5):
            filtered_df = df[(df['Item1'] == row['Item_Key']) | (df['Item2'] == row['Item_Key'])]
            filtered_df['Co-Occurrence'] = filtered_df.apply(
                lambda r: (r['Item2'] if r['Item1'] == row['Item_Key'] else r['Item1'], r['Count']), axis=1)
            top_items = filtered_df.nlargest(max_items, 'Count')['Co-Occurrence'].tolist()
            return ', '.join([f"{item} ({count})" for item, count in top_items])

        # Apply function to get top co-occurrences
        df_item_keys_sorted['Most_Frequent_Co_Occurrences'] = df_item_keys_sorted.apply(top_co_occurrences, axis=1, df=df_co_occurrences)

        # Selecting and renaming columns for the final table
        final_table = df_item_keys_sorted[['Item_Key', 'Count', 'Most_Frequent_Co_Occurrences']]
        final_table.columns = ['Concept', 'Count', 'Most Frequently Used in Co-Occurrence With']
        
        def add_line_breaks(x):
            return [html.Span(word) if i == len(x.split(', ')) - 1 else html.Span([word, html.Br()]) 
                    for i, word in enumerate(x.split(', '))]

        final_table.loc[:,'Most Frequently Used in Co-Occurrence With'] = final_table['Most Frequently Used in Co-Occurrence With'].apply(add_line_breaks)


        # Add count of distinct users to each concept
        exploded_df = df_qt_query_master_item_keys.explode('item_keys')
        user_counts_per_concept = exploded_df.groupby('item_keys')['user_id'].nunique().reset_index()
        user_counts_per_concept.columns = ['Concept', 'Users']
        final_table = pd.merge(final_table, user_counts_per_concept, on='Concept', how='left')
        final_table = final_table[['Concept', 'Count', 'Users', 'Most Frequently Used in Co-Occurrence With']]
        
        # Calculate total pages
        page_size = 5
        total_pages = (len(final_table) + page_size - 1) // page_size

        # Handle pagination
        start_row = (active_page - 1) * page_size
        end_row = start_row + page_size
        # Get the DataFrame slice for the current page
        final_table = final_table.iloc[start_row:end_row]
        
        # Creating a table using dbc
        table_header = [
            html.Thead(html.Tr([html.Th(col) for col in final_table.columns]))
        ]

        table_body = [
            html.Tbody([
                html.Tr([
                    html.Td(final_table.iloc[i][col]) if col != 'Most Frequently Used in Co-Occurrence With' 
                    else html.Td(final_table.iloc[i][col])
                    for col in final_table.columns
                ]) for i in range(len(final_table))
            ])
        ]
                
        table_concepts = dbc.Row([
            dbc.Col(
                dbc.Table(
                    table_header + table_body,
                    responsive=True,
                    striped=True,
                    bordered=True,
                    hover=True,
                    style={
                        'textAlign': 'left',
                        'padding': '5px',
                        'fontSize': '12px'
                    }
                ),
                md=12
            )
        ])
    else:
        # No data placeholder figure
        table_concepts = no_data_placeholder("Concepts frequently queried together", "No concept queried in co-occurance with other concepts")
        total_pages=0

    return table_concepts, total_pages, active_page

#---------------------------------------------------------------------
# 3. Query Complexity
#---------------------------------------------------------------------
@app.callback(
    [
        Output('complexity', 'children'),
        Output('user-complexity', 'children'),
        Output('pagination-complexity', 'max_value'),
        Output('pagination-complexity', 'active_page')
    ],
    [
        Input('df-qt-query-master-filtered', 'data'),
        Input('pagination-complexity', 'active_page')
    ],
    [
        State('pagination-complexity', 'active_page')
    ]
)
def update_complexity_analytics_concept_touch(df_qt_query_master_filtered, active_page, current_page):
    
    # Pagination functionalities
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'df-qt-query-master-filtered':
        active_page = 1
    else:
        active_page = current_page if current_page else 1

    # Decode and decompress
    df_qt_query_master_filtered = decode_and_decompress_dataframe(df_qt_query_master_filtered)

    # Simple complexity metric
    def get_complexity_score(concept_touch, num_constraints, temporal_query_constraints):
        concept_touch_weight = 1
        constraints_weight = 3
        temporal_query_weight = 5
        complexity_score = (concept_touch * concept_touch_weight) + (num_constraints * constraints_weight) + (temporal_query_constraints*temporal_query_weight)
        return complexity_score

    # Prepare complexity dataframe
    df_qt_query_master_xml_complexity= df_qt_query_master_filtered.copy()
    # Extract and count item_keys (i.e. number of concepts touched per query)
    df_qt_query_master_xml_complexity['item_keys'] = df_qt_query_master_xml_complexity['request_xml'].apply(extract_item_keys)
    df_qt_query_master_xml_complexity['concept_touch'] = df_qt_query_master_xml_complexity['item_keys'].apply(len)
    # Get contraints (date contraints, value constraints, total occurance constraints)
    df_qt_query_master_xml_complexity['num_constraints'] = df_qt_query_master_xml_complexity['request_xml'].apply(count_query_constraint)
    # Get temporal query definitions by subquery contraints
    df_qt_query_master_xml_complexity['temporal_query_constraints'] = df_qt_query_master_xml_complexity['request_xml'].apply(count_temporal_query_constraints)
    # Sum up contraints and temporal query constraints
    df_qt_query_master_xml_complexity['all_query_constraints'] = df_qt_query_master_xml_complexity['num_constraints'] + df_qt_query_master_xml_complexity['temporal_query_constraints']
    # Get complexity score
    df_qt_query_master_xml_complexity['query_complexity'] = get_complexity_score(df_qt_query_master_xml_complexity['concept_touch'], df_qt_query_master_xml_complexity['num_constraints'], df_qt_query_master_xml_complexity['temporal_query_constraints'])

    # Count duplicate queries
    duplicates = df_qt_query_master_filtered.duplicated(subset=['generated_sql'], keep=False) # Do based on SQL, as XML logs query
    duplicate_count = duplicates.sum() 
    
    # Create histogram
    if not df_qt_query_master_filtered.empty:
        # Create figure
        fig = make_subplots(rows=1, cols=3)
        fig.add_trace(go.Histogram(x=df_qt_query_master_xml_complexity['concept_touch'], nbinsx=20, name='Concept Touch'),row=1, col=1)
        fig.add_annotation(
                    text=f"Number of Queries: {len(df_qt_query_master_xml_complexity)}<br>Min: {df_qt_query_master_xml_complexity['concept_touch'].min()}, Max: {df_qt_query_master_xml_complexity['concept_touch'].max()},  Mean: {df_qt_query_master_xml_complexity['concept_touch'].mean().round(2)}, Median: {df_qt_query_master_xml_complexity['concept_touch'].median()}, <br>Number of duplicate queries: {duplicate_count}",
                    xref="x domain",
                    yref="y domain",
                    x=0,
                    y=-0.55,
                    row=1,
                    col=1,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="left"
                )
        fig.add_trace(go.Histogram(x=df_qt_query_master_xml_complexity['all_query_constraints'], nbinsx=20, name='Constraints and Temporal Queries'),row=1, col=2)
        fig.add_annotation(
                    text=f"Number of Queries: {len(df_qt_query_master_xml_complexity)}<br>Min: {df_qt_query_master_xml_complexity['all_query_constraints'].min()}, Max: {df_qt_query_master_xml_complexity['all_query_constraints'].max()},  Mean: {df_qt_query_master_xml_complexity['all_query_constraints'].mean().round(2)}, Median: {df_qt_query_master_xml_complexity['all_query_constraints'].median()}, <br>Number of duplicate queries: {duplicate_count}",
                    xref="x domain",
                    yref="y domain",
                    x=0,
                    y=-0.55,
                    row=1,
                    col=2,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="left"
                )
        fig.add_trace(go.Histogram(x=df_qt_query_master_xml_complexity['query_complexity'], nbinsx=20, name='Query Complexity Score'),row=1, col=3)
        fig.add_annotation(
                    text=f"Number of Queries: {len(df_qt_query_master_xml_complexity)}<br>Min: {df_qt_query_master_xml_complexity['query_complexity'].min()}, Max: {df_qt_query_master_xml_complexity['query_complexity'].max()},  Mean: {df_qt_query_master_xml_complexity['query_complexity'].mean().round(2)}, Median: {df_qt_query_master_xml_complexity['query_complexity'].median()}, <br>Number of duplicate queries: {duplicate_count}",
                    xref="x domain",
                    yref="y domain",
                    x=0,
                    y=-0.55,
                    row=1,
                    col=3,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="left"
                )

        fig.update_layout(
            title_text='Query Complexity',
            xaxis1_title='Concept Touch',
            xaxis2_title='Constraints and Temporal Queries',
            xaxis3_title='Query Complexity Score',
            yaxis_title='Frequency',
            showlegend=False,
            height=400,
            margin=dict(l=40, r=40, t=60, b=120) 
        )

        # Create dash graph element
        complexity_histograms = dbc.Row([
            dbc.Col(
                dcc.Graph(
                    figure=fig
                ),
                md=12
            )
        ])

        #---------------------------------------------------------------------
        # Complexity score by user
        #---------------------------------------------------------------------

        # filter out users with less than 10 queries
        df_qt_query_master_xml_complexity_by_user = df_qt_query_master_xml_complexity.groupby('user_id').filter(lambda x: len(x) > 10)
        
        # get user counts and sort in descending order
        user_ids = df_qt_query_master_xml_complexity_by_user['user_id'].value_counts().index

        # Calculate total pages
        page_size = 3
        total_pages = (len(user_ids) + page_size - 1) // page_size

        # Handle pagination
        start_user = (active_page - 1) * page_size
        end_user = start_user + page_size
        # Get the DataFrame slice for the current page
        user_ids = user_ids[start_user:end_user]

        if len(user_ids) > 0:
            cols = 3
            rows = math.ceil(len(user_ids) / cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles = [f'Query Complexity for {uid}' for uid in user_ids])
            
            #for user_id in user_ids:
            for idx, user_id in enumerate(user_ids):
                row = idx // cols + 1
                col = idx % cols + 1
                user_df = df_qt_query_master_xml_complexity_by_user[df_qt_query_master_xml_complexity_by_user['user_id'] == user_id]

                # count duplicate queries
                duplicates = user_df.duplicated(subset=['generated_sql'], keep=False)
                duplicate_count = duplicates.sum()
            
                fig.add_trace(go.Histogram(x=user_df['query_complexity'], nbinsx=20, name=f'Query Complexity for {user_id}', marker_color=color_palette[2]),row=row, col=col) 
                fig.add_annotation(
                    text=f"Number of Queries: {len(user_df)}<br>Min: {user_df['query_complexity'].min()}, Max: {user_df['query_complexity'].max()},  Mean: {user_df['query_complexity'].mean().round(2)}, Median: {user_df['query_complexity'].median()}<br>Number of duplicate queries: {duplicate_count}",
                    xref="x domain",
                    yref="y domain",
                    x=0,
                    y=-0.55,
                    showarrow=False,
                    row=row, 
                    col=col,
                    font=dict(size=14, color="black"),
                    align="left"
                )

            fig.update_layout(
                title_text='Query Complexity Score by User (for Users with more than 10 Queries)',
                yaxis_title='Frequency',
                height=500,
                margin=dict(l=40, r=40, t=120, b=160),
                showlegend=False
            )
            fig.update_xaxes(title_text='Query Complexity Score')

            user_complexity_histograms = dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=fig
                    ),
                    md=12
                )
            ])
        else:
            user_complexity_histograms = no_data_placeholder("Query Complexity by User (for Users with more than 10 Queries)", "No user with more than 10 queries")
            total_pages = 0
    else:
        # No data placeholder figures
        complexity_histograms = no_data_placeholder("Query Complexity", "No queries in the indicated time period" )
        user_complexity_histograms = no_data_placeholder("Query Complexity by User (for Users with more than 10 Queries)", "No queries in the indicated time period")
        total_pages = 0

    return complexity_histograms, user_complexity_histograms, total_pages, active_page

# if __name__ == '__main__':
#     if LOCAL_ONLY:
#         # for local only deployment
#         app.run_server(debug=True)
#     else:
#         # for docker / prod deployment
#         app.run_server(debug=False, host="0.0.0.0")


if __name__ == '__main__':
    if LOCAL_ONLY:
        # for local only deployment
        app.run_server(debug=False, use_reloader=False, port=8050)
    else:
        # for docker / prod deployment
        app.run_server(debug=False, use_reloader=False, host="0.0.0.0")