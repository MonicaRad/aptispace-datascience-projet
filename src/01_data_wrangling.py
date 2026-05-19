import os, sys
sys.path.append('/home/runner/work/aptispace-datascience-projet/aptispace-datascience-projet')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import sys
import pandas as pd
import numpy as np

# Ajout du dossier parent au chemin de recherche de modules pour importer 'src'
sys.path.append(os.path.abspath('..'))
from src import data_clean as dc

print("Libraries importées avec succès ! Prêt à démarrer le Wrangling.")


raw_data_path = "../data/raw/owid-monkeypox-data.csv"

df_raw = pd.read_csv(raw_data_path)

df_raw.head()


print("Dimensions :", df_raw.shape)
df_raw.info()


print("Valeurs manquantes par colonne :")
df_raw.isnull().sum()


print("Taux de valeurs manquantes (%) :")
(df_raw.isnull().mean() * 100).sort_values(ascending=False)


print("Nombre de doublons :", df_raw.duplicated().sum())


df_raw.describe()


df_clean = df_raw.copy()

cols_to_drop = ['iso_code']

df_clean = df_clean.drop(
    columns=[col for col in cols_to_drop if col in df_clean.columns]
)

df_clean.head()


df_clean = df_clean.rename(columns={
    'location': 'country',
    'new_cases': 'daily_new_cases',
    'new_deaths': 'daily_new_deaths',
    'total_cases': 'total_cases',
    'total_deaths': 'total_deaths'
})

df_clean.head()


df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

print(df_clean['date'].isnull().sum())
df_clean[['date']].head()


countries_to_exclude = [
    "Africa", "Europe", "North America", "South America",
    "Asia", "World", "Oceania", "Puerto Rico"
]

df_clean = df_clean[~df_clean['country'].isin(countries_to_exclude)]

df_clean['country'].unique()[:20]


redundant_cols = [
    'new_cases_smoothed',
    'new_deaths_smoothed',
    'new_cases_per_million',
    'total_cases_per_million',
    'new_cases_smoothed_per_million',
    'new_deaths_per_million',
    'total_deaths_per_million',
    'new_deaths_smoothed_per_million'
]

existing_redundant = [col for col in redundant_cols if col in df_clean.columns]

df_clean = df_clean.drop(columns=existing_redundant)

print("Colonnes supprimées :", existing_redundant)
df_clean.head()


# Colonnes à imputer
cols_to_impute = [
    "total_deaths",
    "daily_new_cases",
    "daily_new_deaths"
]

# Conversion de la date
df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

# Création d'une colonne mois
df_clean["year_month"] = df_clean["date"].dt.to_period("M")

# Imputation par médiane mensuelle par pays, puis fallback
for col in cols_to_impute:
    if col in df_clean.columns:
        # 1. Médiane par pays et par mois
        df_clean[col] = df_clean.groupby(["country", "year_month"])[col].transform(
            lambda x: x.fillna(x.median())
        )

df_clean = df_clean.drop(columns=["year_month"])
# Vérification
print("Valeurs manquantes après imputation :")
print(df_clean[cols_to_impute].isnull().sum())


# Suppression des doublons
df_clean = df_clean.drop_duplicates()

# Tri par pays puis par date
df_clean = df_clean.sort_values(by=['country', 'date'])

# Réinitialisation de l'index
df_clean = df_clean.reset_index(drop=True)

df_clean.head()


print("Dimensions finales :", df_clean.shape)
print("Doublons restants :", df_clean.duplicated().sum())
print("Valeurs manquantes restantes :")
df_clean.isnull().sum()


processed_path = "../data/processed/owid-monkeypox-data_clean.csv"

df_clean.to_csv(processed_path, index=False)

print(f"Données nettoyées sauvegardées dans : {processed_path}")

