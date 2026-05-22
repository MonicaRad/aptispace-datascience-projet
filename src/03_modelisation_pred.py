import os, sys
sys.path.append('C:\Users\Monica\aptispace-datascience-projet')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath('..'))

from src import utils_viz as uv

uv.set_custom_style()

print("Librairies de modélisation prêtes.")


df = pd.read_csv('../data/processed/owid-monkeypox-data_clean.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['country', 'date']).reset_index(drop=True)

# Choix du pays à modéliser
# Ici on prend le pays avec le plus de nouveaux cas sur toute la période
selected_country = (
    df.groupby("country")["daily_new_cases"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

print("Pays sélectionné :", selected_country)

df_country = df[df["country"] == selected_country].copy()
df_country = df_country.sort_values("date").reset_index(drop=True)

df_country.head()


df_feat = df_country.copy()

# Variables temporelles
df_feat["year"] = df_feat["date"].dt.year
df_feat["month"] = df_feat["date"].dt.month
df_feat["dayofweek"] = df_feat["date"].dt.dayofweek

# Lags courts et longs
for lag in [1, 2, 3, 7, 14, 21]:
    df_feat[f"daily_new_cases_lag{lag}"] = df_feat["daily_new_cases"].shift(lag)

# Lags décès
df_feat["daily_new_deaths_lag1"] = df_feat["daily_new_deaths"].shift(1)
df_feat["daily_new_deaths_lag7"] = df_feat["daily_new_deaths"].shift(7)

# Variables cumulées retardées
df_feat["total_cases_lag1"] = df_feat["total_cases"].shift(1)
df_feat["total_deaths_lag1"] = df_feat["total_deaths"].shift(1)

# Moyennes mobiles basées uniquement sur le passé
for window in [3, 7, 14, 21]:
    df_feat[f"rolling_mean_{window}"] = (
        df_feat["daily_new_cases"]
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
    )

    df_feat[f"rolling_max_{window}"] = (
        df_feat["daily_new_cases"]
        .shift(1)
        .rolling(window=window, min_periods=1)
        .max()
    )

# Variation récente
df_feat["diff_lag1_lag7"] = (
    df_feat["daily_new_cases_lag1"] - df_feat["daily_new_cases_lag7"]
)

df_feat["diff_lag7_lag14"] = (
    df_feat["daily_new_cases_lag7"] - df_feat["daily_new_cases_lag14"]
)

# Suppression des NaN générés par les lags
df_feat = df_feat.dropna().reset_index(drop=True)

df_feat.head()


target = "daily_new_cases"

features = [
    "year",
    "month",
    "dayofweek",
    "daily_new_cases_lag1",
    "daily_new_cases_lag2",
    "daily_new_cases_lag3",
    "daily_new_cases_lag7",
    "daily_new_cases_lag14",
    "daily_new_cases_lag21",
    "daily_new_deaths_lag1",
    "daily_new_deaths_lag7",
    "total_cases_lag1",
    "total_deaths_lag1",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_21",
    "rolling_max_3",
    "rolling_max_7",
    "rolling_max_14",
    "rolling_max_21",
    "diff_lag1_lag7",
    "diff_lag7_lag14"
]

split_index = int(len(df_feat) * 0.8)

train = df_feat.iloc[:split_index]
test = df_feat.iloc[split_index:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print(f"Taille Train : {X_train.shape}")
print(f"Taille Test  : {X_test.shape}")
print("Date min train :", train["date"].min())
print("Date max train :", train["date"].max())
print("Date min test  :", test["date"].min())
print("Date max test  :", test["date"].max())


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "Modèle": name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    }

# Baseline naïve : nouveaux cas de la veille
y_baseline = test["daily_new_cases_lag1"].values

metrics = []
predictions = {}

metrics.append(evaluate_model("Baseline naïve", y_test, y_baseline))
predictions["Baseline naïve"] = y_baseline

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=6,
        min_samples_leaf=3,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=6,
        min_samples_leaf=3,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42
    )
}

for name, model in models.items():
    # Transformation log pour stabiliser la cible
    y_train_log = np.log1p(y_train)

    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # Sécurité : pas de valeurs négatives
    y_pred = np.clip(y_pred, 0, None)

    # Arrondi car on prédit un nombre de cas
    y_pred = np.round(y_pred).astype(int)

    predictions[name] = y_pred
    metrics.append(evaluate_model(name, y_test, y_pred))

results_metrics = pd.DataFrame(metrics).sort_values(by="MAE")
results_metrics


best_model_name = results_metrics.iloc[0]["Modèle"]
best_y_pred = predictions[best_model_name]

print("Meilleur modèle :", best_model_name)
print(results_metrics)


results_country = test[["date"]].copy()
results_country["Valeurs réelles"] = y_test.values
results_country["Prédictions"] = best_y_pred

results_plot = results_country.sort_values("date").head(30)

results_long = results_plot.melt(
    id_vars=["date"],
    value_vars=["Valeurs réelles", "Prédictions"],
    var_name="type",
    value_name="daily_new_cases"
)

plt.figure(figsize=(14, 6))

sns.barplot(
    data=results_long,
    x="date",
    y="daily_new_cases",
    hue="type"
)

plt.title(
    f"Comparaison quotidienne des nouveaux cas réels et prédits - {selected_country} ({best_model_name})"
)
plt.xlabel("Date")
plt.ylabel("Nouveaux cas quotidiens")
plt.xticks(rotation=90)
plt.legend(title="")
plt.tight_layout()
plt.show()

