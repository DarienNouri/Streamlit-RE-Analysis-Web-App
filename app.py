import streamlit as st
import pandas as pd
import plotly.express as px
import plotlyFunctions
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
@st.cache
def loadData():
    df = pd.read_csv('reDataSample.csv', encoding='utf-8')

    return df
df = loadData()
@st.cache
def prepData(df):

    df = df[df['unformattedPrice'] < 40000000]
    df = df[df['zestimate'] < 40000000]
    df = df[df['baths'] < 25]
    df = df[df['beds'] < 40]
    df = df[df['area'] < 25000]
    return df

@st.cache
def makeList(df):
    statesDf = df['addressState'].unique()
    statesList = ['All States']
    for i in statesDf:
        statesList.append(i)
    return statesList



df = prepData(df)
statesList = makeList(df)
statesCount = df['addressState'].value_counts()



#%%Start Streamlit


def main():
    st.subheader("By: Darien Nouri")
    st.title("Exploratory Analysis of Acquired Real Estate Data ")
    col1, col2, = st.columns((10, 10))

    px.set_mapbox_access_token(
        "pk.eyJ1IjoiZG5vdXJpMTgiLCJhIjoiY2w0bnZraTVtMDE1dTNibDZ3cmNjcGphZSJ9.wttGjVxiFNcQQcfbWxaw8Q")
    state1 = col1.selectbox("Select a State:", statesList)
    state2 = col2.selectbox("Select a State2:", statesList[1:])

    if state1 == 'All States':
        dfState1 = df
    else:
        dfState1 = df.loc[df['addressState'] == state1]
    if state2 == 'All States':
        dfState2 = df
    else:
        dfState2 = df.loc[df['addressState'] == state2]
    ##C:\Users\darie\OneDrive - nyu.edu\PersonalProjects\Zillow_For_resume\streamlit\app.py

    lon = dfState1['longitude']
    lat = dfState1['latitude']

    zoom1, center1 = plotlyFunctions.zoom_center(lons=lon, lats=lat)
    print(zoom1)
    print(center1)
    print()
    zoomFun1, centerFun1 = plotlyFunctions.get_plotting_zoom_level_and_center_coordinates_from_lonlat_tuples(
        longitudes=lon, latitudes=lat)
    print(zoomFun1)
    print(centerFun1)
    cen1 = {'lon': centerFun1[0], 'lat': centerFun1[1]}

    figCol1 = px.scatter_mapbox(dfState1, lat="latitude", lon="longitude", color="unformattedPrice",
                                size="unformattedPrice",
                                color_continuous_scale=px.colors.sequential.Plasma, title=state1, zoom=zoomFun1,
                                center=cen1)
    # %%
    # st.header("Wealth Concentrations by Property Listing Prices Per US State")
    figCol1.update_layout(height=800)
    col1.plotly_chart(figCol1, height=800, use_container_width=True)

    col1.header = "Correlation Heatmap"

    corFig = plotlyFunctions.makeHeatmap(dfState1[
                                             ['unformattedPrice', 'zestimate', 'taxAssessedValue', 'addressZipcode',
                                              'beds', 'baths', 'area', 'latitude', 'longitude']])
    col1.plotly_chart(corFig, height=800, use_container_width=True)

    columnnsList = df[
        ['unformattedPrice', 'zestimate', 'taxAssessedValue', 'addressZipcode', 'beds', 'baths', 'area', 'latitude',
         'longitude']].columns.tolist()

    independent1 = col1.selectbox("Select a Dependent Varaible for Model Train:", columnnsList)
    dependent1 = col1.selectbox("Select a Independent Varaible for Model Train:", columnnsList)

    figOls = px.scatter(dfState1, x=independent1, y=dependent1, trendline="ols", color='unformattedPrice')
    figOls.update_layout(height=750)
    col1.plotly_chart(figOls, height=750, use_container_width=True)
    results = px.get_trendline_results(figOls)
    statsSummary = results.iloc[0][0].summary()
    col1.text(statsSummary)

    # dfCorr = df[['unformattedPrice','zestimate','addressStreet','addressCity', 'addressZipcode','latitude', 'longitude','beds', 'baths', 'area','lotAreaValue']]

    if state2 == 'All States':
        dfState2 = df
    else:
        dfState2 = df.loc[df['addressState'] == state2]
    ##C:\Users\darie\OneDrive - nyu.edu\PersonalProjects\Zillow_For_resume\streamlit\app.py

    lon2 = dfState2['longitude']
    lat2 = dfState2['latitude']

    zoom2, center2 = plotlyFunctions.zoom_center(lons=lon2, lats=lat2)

    zoomFun2, centerFun2 = plotlyFunctions.get_plotting_zoom_level_and_center_coordinates_from_lonlat_tuples(
        longitudes=lon2, latitudes=lat2)
    cen2 = {'lon': centerFun2[0], 'lat': centerFun2[1]}

    figCol2 = px.scatter_mapbox(dfState2, lat="latitude", lon="longitude", color="unformattedPrice",
                                size="unformattedPrice",
                                color_continuous_scale=px.colors.sequential.Plasma, title=state2, zoom=zoomFun2,
                                center=cen2)
    # %%
    # st.header("Wealth Concentrations by Property Listing Prices Per US State")
    figCol2.update_layout(height=800)
    col2.plotly_chart(figCol2, height=800, use_container_width=True)

    col2.header = "Correlation Heatmap"

    corFig2 = plotlyFunctions.makeHeatmap(dfState2[
                                              ['unformattedPrice', 'zestimate', 'taxAssessedValue', 'addressZipcode',
                                               'beds', 'baths', 'area', 'latitude', 'longitude']])
    col2.plotly_chart(corFig2, height=800, use_container_width=True)
    columnnsList2 = df[
        ['unformattedPrice', 'zestimate', 'taxAssessedValue', 'addressZipcode', 'beds', 'baths', 'area', 'latitude',
         'longitude']].columns.tolist()
    independent2 = col2.selectbox("Select a Dependent Varaible for Right Col Model Train:", columnnsList2)
    dependent2 = col2.selectbox("Select a Independent Varaible for Right Col Model Train:", columnnsList2)
    figOls2 = px.scatter(dfState2, x=independent2, y=dependent2, trendline="ols", color='unformattedPrice')
    figOls2.update_layout(height=750)
    col2.plotly_chart(figOls2, height=750, use_container_width=True)
    results = px.get_trendline_results(figOls2)
    statsSummary = results.iloc[0][0].summary()
    col2.text(statsSummary)

    multiVarPredictors = col1.multiselect(label="Select predictors for mutiple regression", options=columnnsList2)

    col1.write(multiVarPredictors)

    figOls2MultiVar = px.scatter(dfState2, x=multiVarPredictors, y='unformattedPrice', trendline="ols",
                                 color='unformattedPrice')


if __name__ == "__main__":
    main()
