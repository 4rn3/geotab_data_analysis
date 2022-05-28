"""
feature engineering + EDA in andere file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import plotly.express as px
import pycountry

@st.cache
def load_data():
    """
    loading data
    """
    return pd.read_csv('./data/final_df.csv').rename({'Lat': 'longitude', 'Lon': 'latitude'}, axis=1)

def weekday_entries(dataframe: pd.DataFrame):
    """
    plotting entries per day of week
    """
    plt.figure(figsize=(25, 15))
    fig = sns.countplot(x="DayOfWeek", data=dataframe, palette="mako")
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    fig.set_title("entries for per day of week")
    return fig.figure

def monthly_entries(dataframe: pd.DataFrame):
    """
    plotting entries per month
    """
    plt.figure(figsize=(25, 15))
    fig = sns.countplot(x=dataframe['Months'], data=dataframe, palette="mako")
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    fig.set_title("entries per month")
    return fig.figure

def days_per_month_in_ds(df: pd.DataFrame):
    """
    Function to calculate the days per month in the dataframe
    """
    days , months = [], []

    for month in df.Months.unique():
        months.append(month)
        days.append(len(df[df['Months'] == month]['Date'].value_counts()))

    fig = plt.figure(figsize=(20,15))
    plt.plot(months, days)
    plt.xlabel('Months')
    plt.ylabel('Days per month')

    return fig.figure


def feature_engineer_covid(dataframe: pd.DataFrame):
    """
    feature engieer the needed features in the covid dataset
    """
    dataframe['Total Confirmed'] = dataframe[['Confirmed']].sum(axis=1)
    dataframe.drop(['Recovered', 'Confirmed', 'Deaths'], axis=1, inplace=True)

    return dataframe

def plot_corona_cases(dataframe: pd.DataFrame):
    """
    plotting the corona cases
    """
    c = dataframe.groupby(['Date']).sum().plot(figsize=(20, 15))
    c.set_xlabel('Month')
    c.set(xlabel=None)
    c.set_ylabel('# Cases')
    plt.axvline(x="2020-12-01", color='red', label='December', linestyle='--')
    plt.grid(linestyle='--', linewidth=0.3)
    plt.legend(bbox_to_anchor=(1.0, 0.4))
    return c.figure

def plot_entries_to_x(dataframe: pd.DataFrame, location: str):
    """
    function to plot the entries for location X
    df: dataframe
    location: airport, city, state or country category from the dataframe
    """
    plt.figure(figsize=(25, 15))
    fig = sns.countplot(x=location, data=dataframe, palette="mako")
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    fig.set_title(f"entries per {location}")
    return fig.figure

def get_airport_coords(dataframe: pd.DataFrame):
    """
    extract coordinates from the dataframe
    """
    lon = dataframe['longitude'].unique()
    lat = dataframe['latitude'].unique()
    return lat, lon

@st.cache
def plot_airports(df: pd.DataFrame):
    """
    plotting the airports
    """

    lon_can, lat_can = get_airport_coords(df[df['Country'] == 'Canada'])
    lon_usa, lat_usa = get_airport_coords(
        df[df['Country'] == 'United States of America (the)'])
    lon_aus, lat_aus = get_airport_coords(df[df['Country'] == 'Australia'])
    lon_chi, lat_chi = get_airport_coords(df[df['Country'] == 'Chile'])

    plt.figure(figsize=(25, 20))

    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcountries()
    m.drawstates()
    m.drawcoastlines()
    m.bluemarble()
    # m.fillcontinents(color='#04BAE3', lake_color='#FFFFFF')
    m.drawparallels(np.arange(-90., 91., 30.))
    m.drawmeridians(np.arange(-180., 181., 60.))

    x, y = m(lat_can, lon_can)
    m.plot(x, y, 'ro', markersize=10, alpha=.6, label='Canada')

    x, y = m(lat_usa, lon_usa)
    m.plot(x, y, 'go', markersize=10, alpha=.6, label='USA')

    x, y = m(lat_aus, lon_aus)
    m.plot(x, y, 'bo', markersize=10, alpha=.6, label='Australia')

    x, y = m(lat_chi, lon_chi)
    m.plot(x, y, 'yo', markersize=10, alpha=.6, label='Chile')

    m.drawmapboundary(fill_color='#FFFFFF')
    plt.legend(loc='best')
    plt.savefig('./pics/airports.png')

def mean_percent_of_baseline_per_airport(dataframe: pd.DataFrame, airport: str):
    """
    function that calculates the mean percent of baseline per airport
    df: dataframe
    airport: specific airport (i.e. 'Kingsford Smith')
    """
    return dataframe['PercentOfBaseline'][dataframe['AirportName'] == airport].mean()

def plot_percent_of_baseline_per_airport(dataframe: pd.DataFrame):
    """
    Plotting the mean percent of baseline per airport
    """
    mean_percent_per_airport = {}
    for airport in dataframe['AirportName'].unique():
        mean_percent_per_airport[airport] = mean_percent_of_baseline_per_airport(
            dataframe, airport)

    plt.figure(figsize=(20, 6))
    fig, _ = plt.subplots()
    fig = sns.barplot(x=np.array(list(mean_percent_per_airport.keys())), y=np.array(
        list(mean_percent_per_airport.values())), data=dataframe, palette="mako")
    fig.set_xticklabels(fig.get_xticklabels(), rotation=75)
    return fig.figure

def plot_heatmap(dataframe: pd.DataFrame):
    """
    Plotting the percent of baseline on map view
    """
    fig = px.scatter_mapbox(dataframe,
                            lat="latitude",
                            lon="longitude",
                            hover_name="AirportName",
                            hover_data=["PercentOfBaseline"],
                            color="PercentOfBaseline",
                            zoom=1,
                            height=600,
                            size="PercentOfBaseline",
                            size_max=30,
                            opacity=0.4,
                            width=1300)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="Mean of trafic on sunday")
    return fig

def plot_covid_map():
    """
    Plot severity of covid per country
    """
    URL_DATASET = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
    df1 = pd.read_csv(URL_DATASET)
    df1.Date = pd.to_datetime(df1.Date)
    df1 = df1[df1.Date.dt.year > 2021]

    df1 = df1[df1.Country.isin(['Australia', 'Canada', 'US', 'Chile'])]
    list_countries = df1['Country'].unique().tolist()


    d_country_code = {}  
    for country in list_countries:
        try:
            country_data = pycountry.countries.search_fuzzy(country)

            country_code = country_data[0].alpha_3
            d_country_code.update({country: country_code})
        except:
            print('could not add ISO 3 code for ->', country)
            d_country_code.update({country: ' '})

    for k, v in d_country_code.items():
        df1.loc[(df1.Country == k), 'iso_alpha'] = v

    fig = px.choropleth(data_frame = df1,
                        locations= "iso_alpha",
                        color= "Confirmed",  
                        hover_name= "Country",
                        color_continuous_scale= 'RdYlGn')

    return fig

def sorted_impact_on_airports(df: pd.DataFrame, amount: int = -1):
    """
    returns sorted dictionary based on the PercentOfBaseline
    amount: amount of airports to show
    """
    mean_percent_per_airport = {}
    for airport in df['AirportName'].unique():
        mean_percent_per_airport[airport] = mean_percent_of_baseline_per_airport(
            df, airport)
    return [(key, value) for (key, value) in sorted(mean_percent_per_airport.items(), key=lambda x: x[1])][:amount]

def main():
    """
    main function
    """
    st.set_option('deprecation.showPyplotGlobalUse', False)

    df = load_data()
    df_covid = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv', parse_dates=['Date'])
    df_covid = feature_engineer_covid(df_covid)

    st.title('Data analysis geotab')
    st.markdown(
        "In this dashboard we'll look at the impact covid had on the airports. ")
    st.markdown(
        "The usage of plotly charts and interactive graphs takes a little while to load initially (be patient).")

    st.markdown("Choose which page you want to see in the sidebar")
    st.markdown("""---""")

    page = st.sidebar.radio(
        '', ('Time related data', 'Location related data', 'Covid related data'))

    if page == 'Time related data':
        st.subheader('Time related analysis')

        time_period = st.radio(
            "What time period would you like to see for the entries", ('per day of week', 'per month'))

        if time_period == 'per month':
            months = st.multiselect('Which month(s) would you like to see', ('March', 'April',
                                  'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))
            month_plot = monthly_entries(df)
            if len(months) != 0:
                selected_months = df[df['Months'].isin(months)]
                month_plot = monthly_entries(selected_months)
            else:
                month_plot = monthly_entries(df)

            st.write("Amount of entries per month")
            st.pyplot(month_plot)

            covid_display = st.checkbox(
                "Would you like to see the covid data?")
            if covid_display:
                st.pyplot(plot_corona_cases(df_covid))

            day_display = st.checkbox(
                "Would you like to see the amount of days per month?")
            if day_display:
                st.pyplot(days_per_month_in_ds(df))

        else:
            days = st.multiselect('Which day(s) would you like to see', ('Monday',
                                'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))

            if len(days) != 0:
                selected_days = df[df['DayOfWeek'].isin(days)]
                day_of_week_plot = weekday_entries(selected_days)
            else:
                day_of_week_plot = weekday_entries(df)


            st.write("Amount of entries per day of week")
            st.pyplot(day_of_week_plot)

    if page == 'Location related data':
        st.subheader('Location related analysis')
        st.write("All airports")
        st.map(df)

        location = st.radio("What location would you like to see",
                            ('Airports', 'Cities', 'States', 'Countries'))

        if location == 'Airports':
            st.write("Entries per airport")

            airport_list = df.AirportName.unique().tolist()
            airports = st.multiselect("What airports would you like to see", (airport_list))

            if len(airports) != 0:
                selected_airports = df[df['AirportName'].isin(airports)]
                airports = plot_entries_to_x(selected_airports, 'AirportName')
            else:
                airports = plot_entries_to_x(df, 'AirportName')

            st.pyplot(airports)

        if location == 'Cities':
            st.write("Entries per city")

            city_list = df.City.unique().tolist()
            cities = st.multiselect("What cities would you like to see", (city_list))

            if len(cities) != 0:
                selected_cities = df[df['City'].isin(cities)]
                cities = plot_entries_to_x(selected_cities, 'City')
            else:
                cities = plot_entries_to_x(df, 'City')

            st.pyplot(cities)

        if location == 'States':
            st.write("Entries per state")

            state_list = df.State.unique().tolist()
            states = st.multiselect("What states would you like to see", (state_list))

            if len(states) != 0:
                selected_states = df[df['State'].isin(states)]
                states = plot_entries_to_x(selected_states, 'State')
            else:
                states = plot_entries_to_x(df, 'State')

            st.pyplot(states)

        if location == 'Countries':
            st.write("Entries per country")

            country_list = df.Country.unique().tolist()
            countries = st.multiselect("What countries would you like to see", (country_list))

            if len(countries) != 0:
                selected_countries = df[df['Country'].isin(countries)]
                country = plot_entries_to_x(selected_countries, 'Country')
            else:
                country = plot_entries_to_x(df, 'Country')

            st.pyplot(country)

            show_airports = st.checkbox(
                "Would you like to see the airports per country?")

            if show_airports:
                plot_airports(df)
                st.image('./pics/airports.png',
                         caption='Airports per country', use_column_width=True)

    if page == 'Covid related data':
        st.subheader('Covid related analysis')

        charts = st.radio("What type of chart would you like",
                          ('bar', 'heatmap'))

        if charts == 'bar':

            impact = st.checkbox(
                "Would you only like to see the most and least impacted airports?")

            if impact:
                least = sorted_impact_on_airports(df, 1)[0][0]
                most = sorted_impact_on_airports(df)[-1][0]

                st.markdown(
                    f"These are the least and most impacted airports, {most} and {least}")

                least_most_impact = df[(df['AirportName'] == least)
                                       | (df['AirportName'] == most)]

                st.plotly_chart(plot_percent_of_baseline_per_airport(least_most_impact))
                st.markdown(
                    f'The least impacted airport is {least} and the most impacted is {most}')

                st.plotly_chart(plot_covid_map())
          
            else:
                percent_of_baseline = plot_percent_of_baseline_per_airport(df)
                st.pyplot(percent_of_baseline)

        if charts == "heatmap":

            impact = st.checkbox(
                "Would you like to only see the most and least impacted airports?")

            if impact:
                least = sorted_impact_on_airports(df, 1)[0][0]
                most = sorted_impact_on_airports(df)[-1][0]

                st.markdown(
                    f"These are the least and most impacted airports, {most} and {least}")

                least_most_impact = df[(df['AirportName'] == least)
                                       | (df['AirportName'] == most)]

                st.plotly_chart(plot_heatmap(least_most_impact))
                st.markdown(
                    f'We see that the least impacted airport is {least} and the most impacted is {most}')
            else:
                st.plotly_chart(plot_heatmap(df))

        st.markdown(
                    "Covid cases per country.")
        st.plotly_chart(plot_covid_map())

if __name__ == "__main__":
    main()
