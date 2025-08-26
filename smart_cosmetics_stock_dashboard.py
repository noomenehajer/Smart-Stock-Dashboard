# smart_cosmetics_stock_dashboard.py
# ------------------------------------------------------------
# SmartCosmeticsStock: Gestion intelligente & durable des stocks
# Dashboard Streamlit + IA (prévision) + analyse énergétique
# ------------------------------------------------------------
# Installation (terminal):
#   pip install -r requirements.txt
# Lancement:
#   streamlit run smart_cosmetics_stock_dashboard.py
#
# Notes:
# - Le script génère des données factices réalistes si aucun fichier n'est fourni.
# - Vous pouvez exporter les datasets depuis la barre latérale.
# - Les prédictions utilisent scikit-learn (RandomForestRegressor).
# - Aucune dépendance lourde (Prophet/ARIMA) pour garder l'installation simple.
# ------------------------------------------------------------
import textwrap
import os
import io
import base64
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------
# Configuration & constantes
# -------------------------------
st.set_page_config(
    page_title="SmartCosmeticsStock",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

np.random.seed(42)

PRODUCT_CATEGORIES = {
    "Parfum": {"temp_opt": 20, "light_sensitive": True, "shelf_life_days": 720, "volume_l": 0.4},
    "Shampooing": {"temp_opt": 22, "light_sensitive": False, "shelf_life_days": 540, "volume_l": 0.6},
    "Crème": {"temp_opt": 20, "light_sensitive": True, "shelf_life_days": 540, "volume_l": 0.25},
    "Sérum": {"temp_opt": 18, "light_sensitive": True, "shelf_life_days": 365, "volume_l": 0.08},
    "Déodorant": {"temp_opt": 22, "light_sensitive": False, "shelf_life_days": 540, "volume_l": 0.3},
    "Gel Douche": {"temp_opt": 22, "light_sensitive": False, "shelf_life_days": 540, "volume_l": 0.7},
}

WAREHOUSE_ZONES = {
    "Ambiante": {"setpoint": 24, "base_kwh_m3_day": 0.15, "lighting_kwh_h": 1.1},
    "Fraîche": {"setpoint": 20, "base_kwh_m3_day": 0.22, "lighting_kwh_h": 1.2},
    "Froide": {"setpoint": 8,  "base_kwh_m3_day": 0.45, "lighting_kwh_h": 1.4},
}

OUTSIDE_TEMP_PROFILE = {
    1: 12,  2: 14, 3: 16, 4: 18, 5: 22, 6: 28,
    7: 32,  8: 33, 9: 27, 10: 22, 11: 17, 12: 13
}

# -------------------------------
# Helpers
# -------------------------------
def month_name_fr(m: int) -> str:
    return ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"][m-1]

REAL_PRODUCTS = {
    "Parfum": [
        ("Chanel No.5", "Chanel"),
        ("Dior Sauvage", "Dior"),
        ("Gucci Bloom", "Gucci"),
        ("Yves Saint Laurent Libre", "YSL"),
        ("Tom Ford Black Orchid", "Tom Ford"),
        ("Jo Malone Peony & Blush Suede", "Jo Malone")
    ],
    "Shampooing": [
        ("Head & Shoulders Classic", "Procter & Gamble"),
        ("L'Oréal Elvive", "L'Oréal"),
        ("Dove Daily Moisture", "Dove"),
        ("Pantene Repair & Protect", "Pantene"),
        ("Herbal Essences Bio:Renew", "Herbal Essences"),
        ("Bumble and Bumble Hairdresser's Invisible Oil", "Bumble and Bumble")
    ],
    "Crème": [
        ("Nivea Soft Moisturizing Cream", "Nivea"),
        ("La Roche-Posay Toleriane Ultra", "La Roche-Posay"),
        ("Clinique Moisture Surge", "Clinique"),
        ("Neutrogena Hydro Boost", "Neutrogena"),
        ("Eucerin Advanced Repair", "Eucerin"),
        ("Aveeno Daily Moisturizing", "Aveeno")
    ],
    "Sérum": [
        ("Estée Lauder Advanced Night Repair", "Estée Lauder"),
        ("The Ordinary Hyaluronic Acid", "The Ordinary"),
        ("L'Oréal Paris Revitalift", "L'Oréal"),
        ("Kiehl's Midnight Recovery", "Kiehl's"),
        ("Vichy Mineral 89", "Vichy"),
        ("Caudalie Vinoperfect", "Caudalie")
    ],
    "Déodorant": [
        ("Dove Original", "Dove"),
        ("Nivea Men Fresh Active", "Nivea"),
        ("Old Spice High Endurance", "Old Spice"),
        ("Rexona Clinical", "Rexona"),
        ("Secret Aluminum Free", "Secret"),
        ("Mitchum Clean Control", "Mitchum")
    ],
    "Gel Douche": [
        ("Neutrogena Rainbath", "Neutrogena"),
        ("Dove Deep Moisture", "Dove"),
        ("Nivea Creme Soft", "Nivea"),
        ("L'Oréal Paris Men Expert", "L'Oréal"),
        ("Irish Spring Original", "Irish Spring"),
        ("Aveeno Daily Moisturizing", "Aveeno")
    ]
}

##def gen_products(n_per_cat: int = 6) -> pd.DataFrame:
##    rows = []
##    pid = 1
##    for cat, meta in PRODUCT_CATEGORIES.items():
##        for i in range(n_per_cat):
##            name = f"{cat} {chr(65+i)}"
##            shelf = meta["shelf_life_days"]
##            rows.append({
##                "product_id": pid,
##                "product_name": name,
##                "category": cat,
##                "temp_opt": meta["temp_opt"],
##                "light_sensitive": meta["light_sensitive"],
##                "shelf_life_days": shelf,
##                "unit_volume_l": meta["volume_l"],
##                "unit_value": np.round(np.random.uniform(8, 80), 2)
##            })
##            pid += 1
##    return pd.DataFrame(rows)
def gen_products(n_per_cat: int = 6) -> pd.DataFrame:
    rows = []
    pid = 1
    for cat, meta in PRODUCT_CATEGORIES.items():
        product_list = REAL_PRODUCTS.get(cat, [])
        # Limit to minimum of available products or n_per_cat
        count = min(len(product_list), n_per_cat)
        for i in range(count):
            name, brand = product_list[i]
            shelf = meta["shelf_life_days"]
            rows.append({
                "product_id": pid,
                "product_name": name,
                "brand": brand,
                "category": cat,
                "temp_opt": meta["temp_opt"],
                "light_sensitive": meta["light_sensitive"],
                "shelf_life_days": shelf,
                "unit_volume_l": meta["volume_l"],
                "unit_value": np.round(np.random.uniform(8, 80), 2)
            })
            pid += 1
    return pd.DataFrame(rows)


def seasonal_factor(day: pd.Timestamp, category: str) -> float:
    # Simple saisonnalité: parfums en hiver/fêtes, solaires en été (ici shampoing/gels montent été)
    month = day.month
    f = 1.0
    if category == "Parfum":
        if month in [11, 12, 1]: f *= 1.25
    if category in ["Shampooing", "Gel Douche"]:
        if month in [6, 7, 8]: f *= 1.2
    if category in ["Crème", "Sérum"]:
        if month in [1, 2, 12]: f *= 1.15
    return f


def gen_daily_data(products: pd.DataFrame, start: str = None, days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    dates = pd.date_range(start=start, periods=days, freq="D")

    inv_rows, sales_rows = [], []
    for _, p in products.iterrows():
        # base demand per day
        base = {
            "Parfum": 6, "Shampooing": 12, "Crème": 9, "Sérum": 5, "Déodorant": 8, "Gel Douche": 10
        }[p["category"]]
        # initial stock
        stock = int(np.random.uniform(400, 1400))
        for d in dates:
            demand = np.random.poisson(lam=base * seasonal_factor(d, p["category"]))
            # occasional promo boosts
            promo = 1 if np.random.rand() < 0.06 else 0
            if promo: demand = int(demand * np.random.uniform(1.2, 1.6))
            sales = min(stock, demand)
            stock = max(0, stock - sales + np.random.poisson(lam=base*0.8))  # restock stochastic

            # choose zone by temp_opt
            if p["temp_opt"] <= 10:
                zone = "Froide"
            elif p["temp_opt"] <= 21:
                zone = "Fraîche"
            else:
                zone = "Ambiante"

            inv_rows.append({
                "date": d,
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "zone": zone,
                "stock_units": stock,
                "unit_volume_l": p["unit_volume_l"],
                "unit_value": p["unit_value"]
            })
            sales_rows.append({
                "date": d,
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "sales_units": sales,
                "promo": promo,
            })
    inv = pd.DataFrame(inv_rows)
    sales = pd.DataFrame(sales_rows)
    return inv, sales


def compute_energy(inv: pd.DataFrame, lighting_hours: Dict[str, float], zone_setpoints: Dict[str, float]) -> pd.DataFrame:
    # energy per zone per day based on occupied volume and temperature delta
    df = inv.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.date
    df["month"] = pd.to_datetime(df["date"]).dt.month
    # occupied volume in m3: units * liters / 1000
    df["occupied_m3"] = (df["stock_units"] * df["unit_volume_l"]) / 1000.0

    # Outside temperature by month
    df["outside_temp"] = df["month"].map(OUTSIDE_TEMP_PROFILE)

    energies = []
    for (day, zone), g in df.groupby(["day", "zone"]):
        zone_meta = WAREHOUSE_ZONES[zone]
        setpoint = zone_setpoints.get(zone, zone_meta["setpoint"])
        outside = OUTSIDE_TEMP_PROFILE[pd.Timestamp(day).month]

        # temperature lift (cooling): assume coefficient proportional to delta_T
        delta_T = max(0, outside - setpoint)  # cooling effort
        m3 = g["occupied_m3"].sum()
        base = zone_meta["base_kwh_m3_day"] * m3 * (1 + 0.04 * delta_T)  # 4% per °C delta
        lighting = zone_meta["lighting_kwh_h"] * lighting_hours.get(zone, 8)
        total = base + lighting

        energies.append({"date": pd.Timestamp(day), "zone": zone, "occupied_m3": m3,
                         "delta_T": delta_T, "base_kwh": base, "lighting_kwh": lighting, "total_kwh": total})
    ener = pd.DataFrame(energies).sort_values("date")
    return ener


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["day"] = pd.to_datetime(df["date"]).dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # lag features by product
    df = df.sort_values(["product_id", "date"])
    df["sales_lag1"] = df.groupby("product_id")["sales_units"].shift(1).fillna(0)
    df["sales_lag7"] = df.groupby("product_id")["sales_units"].shift(7).fillna(0)
    df["sales_ma7"] = df.groupby("product_id")["sales_units"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def train_predict_product(sales: pd.DataFrame, product_id: int, horizon_days: int = 30) -> Tuple[pd.DataFrame, float]:
    sdf = sales[sales["product_id"] == product_id].copy()
    sdf = make_features(sdf)
    features = ["promo", "dow", "month", "day", "is_weekend", "sales_lag1", "sales_lag7", "sales_ma7"]
    sdf = sdf.dropna(subset=features + ["sales_units"])
    if len(sdf) < 30:
        return pd.DataFrame(), np.nan

    X = sdf[features]
    y = sdf["sales_units"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)

    # Forecast next N days using last known feature values progression
    last_date = pd.to_datetime(sdf["date"].max())
    rows = []
    last_sales = sdf.iloc[-1]["sales_units"]
    lag1 = sdf.iloc[-1]["sales_units"]
    lag7_series = sdf["sales_units"].tolist()[-7:]
    for i in range(1, horizon_days+1):
        d = last_date + pd.Timedelta(days=i)
        dow = d.dayofweek
        month = d.month
        day = d.day
        is_weekend = 1 if dow >= 5 else 0
        promo = 1 if np.random.rand() < 0.04 else 0
        ma7 = np.mean(lag7_series[-7:]) if len(lag7_series) >= 1 else last_sales

        Xn = pd.DataFrame([{
            "promo": promo, "dow": dow, "month": month, "day": day, "is_weekend": is_weekend,
            "sales_lag1": lag1, "sales_lag7": lag7_series[-1] if lag7_series else 0, "sales_ma7": ma7
        }])
        yhat = max(0, model.predict(Xn)[0])
        rows.append({"date": d, "forecast_units": yhat})
        # update lags
        lag7_series.append(yhat)
        lag1 = yhat
        last_sales = yhat
    fc = pd.DataFrame(rows)
    return fc, mae


def kpi_card(label: str, value, help_text: str = None):
    st.metric(label, value)
    if help_text:
        st.caption(help_text)


def recommend_actions(energy_df: pd.DataFrame, inv_df: pd.DataFrame, zone_setpoints: Dict[str, float], lighting_hours: Dict[str, float]) -> pd.DataFrame:
    recs = []
    # 1) Setpoints: increase setpoint by +1°C where delta_T high and products tolerate
    monthly = energy_df.groupby(["zone"]).agg(total_kwh=("total_kwh", "sum"), avg_delta=("delta_T", "mean")).reset_index()
    for _, r in monthly.iterrows():
        zone = r["zone"]
        delta = r["avg_delta"]
        if delta > 4:
            saving = 0.04 * r["total_kwh"]  # ~4% per °C
            recs.append({
                "Priorité": "Haute",
                "Action": f"Augmenter le setpoint de {zone} de +1°C (si qualité OK)",
                "Impact estimé (kWh/mois)": round(saving/12, 1),
                "Hypothèse": "Réduction de ~4% de l'énergie par °C de delta_T"
            })
    # 2) Éclairage: réduire 1h si > 10h
    for z, hrs in lighting_hours.items():
        if hrs > 10:
            # approx saving = lighting_kwh_h * 1h * 30 jours
            lk = WAREHOUSE_ZONES[z]["lighting_kwh_h"]
            recs.append({
                "Priorité": "Moyenne",
                "Action": f"Réduire l'éclairage en zone {z} de 1h/jour",
                "Impact estimé (kWh/mois)": round(lk * 30, 1),
                "Hypothèse": "Éclairage proportionnel aux heures d'utilisation"
            })
    # 3) Réaffectation des produits sensibles à la lumière vers zones à faible éclairage
    ls = inv_df.merge(pd.DataFrame([{"category": k, "light_sensitive": v["light_sensitive"]} for k, v in PRODUCT_CATEGORIES.items()]),
                      on="category", how="left")
    share_ls = ls[ls["light_sensitive"]].groupby("zone")["stock_units"].sum()
    if not share_ls.empty and share_ls.idxmax() == "Ambiante":
        recs.append({
            "Priorité": "Moyenne",
            "Action": "Réaffecter une partie des produits sensibles à la lumière vers la zone Fraîche",
            "Impact estimé (kWh/mois)": 0.0,
            "Hypothèse": "Réduction des pertes qualité, moins d'éclairage intense"
        })
    # 4) Stock optimal: réduire surstocks > 30% au-dessus du MA7 des ventes
    sales = st.session_state.get("sales_data", None)
    if sales is not None:
        s7 = sales.sort_values("date").groupby("product_id")["sales_units"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        last_stock = inv_df.sort_values("date").groupby("product_id").tail(1)
        cat = last_stock[["product_id","product_name","category","stock_units"]].merge(
            sales.groupby("product_id")["sales_units"].mean().rename("avg_sales"), on="product_id", how="left"
        )
        cat["ratio"] = cat["stock_units"] / (cat["avg_sales"]*14 + 1e-6)
        overs = cat[cat["ratio"] > 1.3].sort_values("ratio", ascending=False).head(10)
        for _, o in overs.iterrows():
            recs.append({
                "Priorité": "Haute",
                "Action": f"Réduire le stock de {o['product_name']} (ratio stock/2sem > 1.3)",
                "Impact estimé (kWh/mois)": round(o["stock_units"]*0.001*20, 2),  # approx : volume -> m3 * 20 kWh/mois
                "Hypothèse": "Moins de volume stocké -> moins de kWh"
            })
    if not recs:
        recs.append({"Priorité": "Info", "Action": "Paramètres actuels efficaces", "Impact estimé (kWh/mois)": 0.0, "Hypothèse": ""})
    return pd.DataFrame(recs)


def df_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# -------------------------------
# APP UI
# -------------------------------
import streamlit as st

st.markdown("<h1 style='text-align: center;'>Dashboard Prédictif et Recommandations pour Distribution</h1>", unsafe_allow_html=True)

#st.title("Dashboard Prédictif et Recommandations pour Distribution")

st.caption("Gestion intelligente et durable des stocks de cosmétiques")

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.write("• Gestion des données •")

    # Data generation
    n_per_cat = st.slider("Produits par catégorie", 3, 12, 6, 1)
    days_hist = st.slider("Jours d'historique", 120, 540, 365, 5)

    if st.button("Générer les données factices"):
        st.session_state["products"] = gen_products(n_per_cat)
        inv, sales = gen_daily_data(st.session_state["products"], days=days_hist)
        st.session_state["inventory_data"] = inv
        st.session_state["sales_data"] = sales
        st.success("Données générées ✔️")

    st.markdown("---")
    st.subheader("🔌 Scénario énergétique")
    lighting_hours = {
        "Ambiante": st.slider("Heures éclairage — Ambiante", 4, 16, 10, 1),
        "Fraîche": st.slider("Heures éclairage — Fraîche", 4, 16, 11, 1),
        "Froide": st.slider("Heures éclairage — Froide", 4, 16, 12, 1),
    }
    zone_setpoints = {
        "Ambiante": st.slider("Setpoint °C — Ambiante", 18, 28, WAREHOUSE_ZONES["Ambiante"]["setpoint"], 1),
        "Fraîche": st.slider("Setpoint °C — Fraîche", 14, 24, WAREHOUSE_ZONES["Fraîche"]["setpoint"], 1),
        "Froide": st.slider("Setpoint °C — Froide", 2, 12, WAREHOUSE_ZONES["Froide"]["setpoint"], 1),
    }

    st.markdown("---")
    if "inventory_data" in st.session_state:
        df_to_csv_download(st.session_state["inventory_data"], "inventory_data.csv", "⬇️ Exporter Inventaire (CSV)")
    if "sales_data" in st.session_state:
        df_to_csv_download(st.session_state["sales_data"], "sales_data.csv", "⬇️ Exporter Ventes (CSV)")


# Initialize data at first run
if "inventory_data" not in st.session_state or "sales_data" not in st.session_state:
    st.session_state["products"] = gen_products(n_per_cat=6)
    inv, sales = gen_daily_data(st.session_state["products"], days=365)
    st.session_state["inventory_data"] = inv
    st.session_state["sales_data"] = sales

products = st.session_state["products"]
inventory = st.session_state["inventory_data"]
sales = st.session_state["sales_data"]

# Energy computation
energy = compute_energy(inventory, lighting_hours, zone_setpoints)

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
total_skus = products.shape[0]
total_units = int(inventory.groupby("product_id")["stock_units"].last().sum())
monthly_energy = energy[energy["date"] >= (energy["date"].max() - pd.Timedelta(days=30))]["total_kwh"].sum()
avg_setpoint = np.mean(list(zone_setpoints.values()))

with col1:
    kpi_card("Nombre de SKU", total_skus, "Produits uniques")
with col2:
    kpi_card("Stock total (unités)", f"{total_units:,}".replace(",", " "), "Dernier jour")
with col3:
    kpi_card("Énergie 30j (kWh)", round(monthly_energy, 1), "Total sur 30 jours")
with col4:
    kpi_card("Setpoint moyen (°C)", round(avg_setpoint, 1), "Moyenne zones")

st.markdown("---")

# -------------------------------
# Onglets
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Stocks & Ventes", "⚡ Énergie", "🔮 Prédictions (IA)", "🛠️ Recommandations", "🗂️ Détails & Export"
])

with tab1:
    st.subheader("Vue d'ensemble des stocks")
    latest = inventory.sort_values("date").groupby(["product_id", "product_name", "category", "zone"]).tail(1)
    colA, colB = st.columns([1.2, 1])
    with colA:
        fig = px.bar(latest.sort_values("stock_units", ascending=False).head(20),
                     x="product_name", y="stock_units", color="category",
                     title="Top 20 stocks par produit (unités)")
        fig.update_layout(xaxis_title="", yaxis_title="Unités")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig2 = px.pie(latest, names="category", values="stock_units", title="Répartition du stock par catégorie")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Ventes quotidiennes (toutes catégories)")
    daily_sales = sales.groupby("date")["sales_units"].sum().reset_index()
    fig3 = px.line(daily_sales, x="date", y="sales_units", title="Ventes quotidiennes (unités)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Détails par catégorie")
    cat = st.selectbox("Catégorie", sorted(PRODUCT_CATEGORIES.keys()))
    cat_sales = sales[sales["category"] == cat].groupby("date")["sales_units"].sum().reset_index()
    fig4 = px.area(cat_sales, x="date", y="sales_units", title=f"Ventes — {cat}")
    st.plotly_chart(fig4, use_container_width=True)


with tab2:
    st.subheader("Consommation énergétique")
    monthly = energy.assign(month=lambda d: pd.to_datetime(d["date"]).dt.month)
    msum = monthly.groupby(["month", "zone"])["total_kwh"].sum().reset_index()
    msum["Mois"] = msum["month"].map(lambda m: month_name_fr(int(m)))
    fig5 = px.bar(msum, x="Mois", y="total_kwh", color="zone", barmode="group",
                  title="Énergie mensuelle par zone (kWh)")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.line(energy, x="date", y="total_kwh", color="zone", title="Énergie quotidienne par zone (kWh)")
    st.plotly_chart(fig6, use_container_width=True)

    st.info("Astuce: Ajustez les *setpoints* et les heures d'éclairage depuis la barre latérale pour simuler des économies d'énergie.")


with tab3:
    st.subheader("Prédictions de la demande (IA)")
    prod_name = st.selectbox("Produit", products["product_name"].tolist())
    prod_id = int(products.loc[products["product_name"] == prod_name, "product_id"].iloc[0])
    fc, mae = train_predict_product(sales, prod_id, horizon_days=30)

    if not fc.empty:
        hist = sales[sales["product_id"] == prod_id][["date", "sales_units"]].rename(columns={"sales_units": "units"})
        hist["type"] = "Historique"
        fcf = fc.rename(columns={"forecast_units": "units"})
        fcf["type"] = "Prévision"
        comb = pd.concat([hist.tail(120), fcf], ignore_index=True)

        fig7 = px.line(comb, x="date", y="units", color="type", title=f"Demande — {prod_name} (MAE test={mae:.2f})")
        st.plotly_chart(fig7, use_container_width=True)

        st.caption("Le modèle RandomForest utilise des caractéristiques calendaires + lags (1j, 7j, MA7).")
    else:
        st.warning("Historique insuffisant pour entraîner une prévision fiable.")


with tab4:
    st.subheader("Recommandations d'optimisation")
    recs = recommend_actions(energy, inventory, zone_setpoints, lighting_hours)
    st.dataframe(recs, use_container_width=True, hide_index=True)
    df_to_csv_download(recs, "recommandations.csv", "⬇️ Exporter recommandations (CSV)")

    st.markdown("#### Scénarios rapides")
    colX, colY, colZ = st.columns(3)
    with colX:
        if st.button("Scénario Éco: +1°C partout"):
            for z in zone_setpoints:
                zone_setpoints[z] += 1
            energy2 = compute_energy(inventory, lighting_hours, zone_setpoints)
            gain = energy["total_kwh"].sum() - energy2["total_kwh"].sum()
            st.success(f"Gain estimé sur la période: {gain:.1f} kWh")
    with colY:
        if st.button("Scénario Lumière: -1h partout"):
            lighting_hours2 = {z: max(4, h-1) for z, h in lighting_hours.items()}
            energy2 = compute_energy(inventory, lighting_hours2, zone_setpoints)
            gain = energy["total_kwh"].sum() - energy2["total_kwh"].sum()
            st.success(f"Gain estimé sur la période: {gain:.1f} kWh")
    with colZ:
        if st.button("Scénario Mixte (+1°C & -1h)"):
            lighting_hours2 = {z: max(4, h-1) for z, h in lighting_hours.items()}
            zone_setpoints2 = {z: v+1 for z, v in zone_setpoints.items()}
            energy2 = compute_energy(inventory, lighting_hours2, zone_setpoints2)
            gain = energy["total_kwh"].sum() - energy2["total_kwh"].sum()
            st.success(f"Gain estimé sur la période: {gain:.1f} kWh")

    st.info("Les gains sont approximatifs (modèle simplifié: +4% énergie par °C de delta_T).")


with tab5:
    st.subheader("Détails des données")
    st.write("**Inventaire (dernier état par produit)**")
    latest_display = inventory.sort_values("date").groupby(["product_id", "product_name", "category", "zone"]).tail(1)
    st.dataframe(latest_display.reset_index(drop=True), use_container_width=True)

    st.write("**Ventes (dernier mois)**")
    last_month = sales[sales["date"] >= (sales["date"].max() - pd.Timedelta(days=30))]
    st.dataframe(last_month.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.download_button(
        label="⬇️ Exporter un README projet (Markdown)",
        data=textwrap.dedent(f"""
        # SmartCosmeticsStock

        **Dashboard Streamlit pour la gestion intelligente & durable des stocks de cosmétiques.**

        ## Installation
        ```bash
        pip install -r requirements.txt
        streamlit run smart_cosmetics_stock_dashboard.py
        ```

        ## Fonctions
        - Données factices réalistes (stocks, ventes).
        - Énergie entrepôt par zone (Ambiante, Fraîche, Froide) avec scénarios (setpoints, éclairage).
        - Prédictions de la demande (RandomForest: lags + features calendaires).
        - Recommandations d'optimisation (setpoints, éclairage, surstocks).
        - Export CSV (inventaire, ventes, recommandations).

        ## Notes
        - Modèle énergétique simplifié: +4% énergie par °C de delta_T (extérieur - setpoint).
        - Dataset généré localement: idéal pour les captures d'écran du rapport.
        """).encode("utf-8"),
        file_name="README_SmartCosmeticsStock.md",
        mime="text/markdown"
    )

st.markdown("---")
st.caption("© 2025 SmartCosmeticsStock — Démo académique (données simulées).")
