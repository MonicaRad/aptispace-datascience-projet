import os, sys
sys.path.append('C:\Users\aklih\rendu-data\aptispace-datascience-projet')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('..'))
from src import data_clean as dc
from src import utils_viz as uv

# Activation de la charte graphique personnalisée du projet
uv.set_custom_style(theme='light')
# %matplotlib inline


# Chargement du dataset propre
df = pd.read_csv('../data/processed/owid-monkeypox-data_clean.csv')
df['date'] = pd.to_datetime(df['date'])
df.head()


df_feat = df.copy()

df_feat['year'] = df_feat['date'].dt.year
df_feat['month'] = df_feat['date'].dt.month
df_feat['year_month'] = df_feat['date'].dt.to_period('M').astype(str)
df_feat['dayofweek'] = df_feat['date'].dt.dayofweek

df_feat.head()


top_5_countries = (
    df_feat.groupby('country')['total_cases']
    .max()
    .sort_values(ascending=False)
    .head(5)
    .index
)

df_top5 = df_feat[df_feat['country'].isin(top_5_countries)]

fig1 = uv.plot_generic_trends(
    df_top5,
    x_col='date',
    y_col='total_cases',
    group_col='country'
)

plt.show()


import plotly.express as px

# On récupère le nombre maximum de cas cumulés par pays
map_data = (
    df_feat.groupby("country", as_index=False)["total_cases"]
    .max()
    .sort_values(by="total_cases", ascending=False)
)

fig = px.choropleth(
    map_data,
    locations="country",
    locationmode="country names",
    color="total_cases",
    hover_name="country",
    color_continuous_scale="Reds",
    title="Répartition mondiale des cas cumulés de mpox par pays"
)

fig.update_layout(
    title_x=0.5,
    geo=dict(showframe=False, showcoastlines=True)
)

fig.show()


import plotly.express as px

# Préparation des données mensuelles par pays
map_time_data = (
    df_feat.groupby(["country", "year_month"], as_index=False)
    .agg({
        "total_cases": "max",
        "daily_new_cases": "sum"
    })
)

# Correction des noms de pays pour Plotly
map_time_data["country_map"] = map_time_data["country"].str.replace("_", " ")

# Conversion du mois en date pour garantir l'ordre chronologique
map_time_data["year_month_date"] = pd.to_datetime(map_time_data["year_month"])

# Tri chronologique
map_time_data = map_time_data.sort_values("year_month_date")

fig = px.choropleth(
    map_time_data,
    locations="country_map",
    locationmode="country names",
    color="total_cases",
    hover_name="country",
    hover_data={
        "total_cases": True,
        "daily_new_cases": True,
        "year_month": True,
        "country_map": False
    },
    animation_frame="year_month",
    color_continuous_scale="Reds",
    title="Propagation du mpox dans le monde au fil du temps"
)

fig.update_layout(
    title_x=0.5,
    geo=dict(
        showframe=False,
        showcoastlines=True
    )
)

fig.show()

