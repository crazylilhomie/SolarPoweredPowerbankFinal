import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_buyer_flag(df):
    col = "Q30_Purchase_Likelihood"
    if col not in df.columns:
        return df, None

    buyer_map = {
        "Definitely will purchase (90-100% likely)": 1,
        "Very likely to purchase (70-89%)": 1,
        "Moderately likely (50-69%)": 0,
        "Somewhat likely (30-49%)": 0,
        "Unlikely (10-29%)": 0,
        "Definitely will NOT purchase (0-9%)": 0,
    }

    df["Buyer"] = df[col].map(buyer_map)
    return df, "Buyer"


def create_price_numeric(df):
    col = "Q31_Max_Price_Willing_To_Pay"
    price_map = {
        "Under $40": 35,
        "$40 - $54": 47,
        "$55 - $69": 62,
        "$70 - $84": 77,
        "$85 - $99": 92,
        "$100 - $119": 110,
        "$120 - $149": 135,
        "$150 - $179": 165,
        "$180 - $199": 190,
        "$200+": 210,
    }
    df["Price_numeric"] = df[col].map(price_map)
    return df, "Price_numeric"


def prepare_classification_data(df, target_col):
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q30_Purchase_Likelihood"]

    df_model = df.dropna(subset=[target_col]).copy()
    X = pd.get_dummies(df_model[feature_cols], drop_first=True)
    y = df_model[target_col].astype(int)
    return X, y, feature_cols, X.columns.tolist()


def prepare_regression_data(df, target_col):
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q31_Max_Price_Willing_To_Pay"]

    df_model = df.dropna(subset=[target_col]).copy()
    X = pd.get_dummies(df_model[feature_cols], drop_first=True)
    y = df_model[target_col].astype(float)
    return X, y, feature_cols, X.columns.tolist()


def crossval_evaluate(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_all, y_pred_all, y_score_all = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        mdl = clone(model)
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(pred)

        if hasattr(mdl, "predict_proba"):
            y_score_all.append(mdl.predict_proba(X_test)[:, 1])
        else:
            y_score_all.append(mdl.decision_function(X_test))

    y_true = pd.concat(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_score = np.concatenate(y_score_all)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }

    final_model = clone(model)
    final_model.fit(X, y)
    return final_model, metrics


# =============================================================================
# UPDATED ASSOCIATION RULES (ONLY Q25, Q27, Q9)
# =============================================================================

def run_association_rules(df):
    cols = ["Q25_Important_Features", "Q27_Concerns", "Q9_Outdoor_Activities"]

    transactions = []
    for _, row in df[cols].iterrows():
        basket = []
        for c in cols:
            if isinstance(row[c], str):
                basket += [f"{c}: {x.strip()}" for x in row[c].split("|")]
        if basket:
            transactions.append(basket)

    if not transactions:
        return None, None

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    freq = apriori(df_trans, min_support=0.05, use_colnames=True)
    if freq.empty:
        return df_trans, None

    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    if rules.empty:
        return df_trans, None

    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)
    return df_trans, rules


# =============================================================================
# CLUSTERING FUNCTIONS (WITH PCA)
# =============================================================================

def prepare_clustering_matrix(df, vars):
    df_c = df.dropna(subset=vars).copy()
    X_c = pd.get_dummies(df_c[vars], drop_first=True)
    return df_c, X_c


def summarize_clusters(df):
    s = df.groupby("Cluster").agg(
        Count=("Cluster", "size"),
        Buyer_Rate=("Buyer", "mean"),
        Avg_Price=("Price_numeric", "mean")
    )
    return s, s["Buyer_Rate"].idxmax()


def recommend_price(df):
    buyers = df[df["Buyer"] == 1]
    if buyers["Price_numeric"].isna().all():
        return None
    med = buyers["Price_numeric"].median()
    return int(round(med / 5) * 5)


def compute_preference_importance(df):
    cols = ["Q25_Important_Features", "Q27_Concerns", "Q9_Outdoor_Activities"]
    items = []

    for c in cols:
        if c in df.columns:
            items += df[c].dropna().astype(str).str.split("|").explode().str.strip().tolist()

    if not items:
        return None

    return pd.Series(items).value_counts()


# =============================================================================
# STREAMLIT SETUP
# =============================================================================

st.set_page_config(page_title="Solar Power Bank Dashboard", layout="wide")
st.title("🔋 Solar Power Bank Launch Analytics Dashboard")

uploaded = st.file_uploader("Upload your survey CSV", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
df, buyer_col = create_buyer_flag(df)
df, price_col = create_price_numeric(df)

tabs = st.tabs([
    "📊 Overview",
    "🤖 Classification",
    "📎 Association Rules",
    "👥 Clustering (3D PCA)",
    "💵 Regression",
    "🧪 Score New Customers",
])


# =============================================================================
# 📊 OVERVIEW (INTERACTIVE)
# =============================================================================

with tabs[0]:

    st.header("📊 Overview — Interactive Plotly Charts")

    col1, col2 = st.columns(2)

    with col1:
        if "Q2_Gender" in df.columns:
            st.subheader("Gender Distribution")
            g = df["Q2_Gender"].value_counts().reset_index()
            g.columns = ["Gender", "Count"]
            fig = px.pie(g, names="Gender", values="Count", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Q3_Location_Type" in df.columns:
            st.subheader("Location Type")
            l = df["Q3_Location_Type"].value_counts().reset_index()
            l.columns = ["Location", "Count"]
            fig = px.pie(l, names="Location", values="Count", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

    if "Q1_Age_Group" in df.columns and "Q30_Purchase_Likelihood" in df.columns:
        st.subheader("Purchase Likelihood vs Age Group")
        ct = pd.crosstab(df["Q1_Age_Group"], df["Q30_Purchase_Likelihood"])
        fig = px.imshow(ct, text_auto=True, color_continuous_scale="OrRd")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 🤖 CLASSIFICATION (WITH 5-FOLD CV)
# =============================================================================

with tabs[1]:

    st.header("🤖 Classification — 5-Fold Cross Validation")

    if buyer_col is None:
        st.warning("Buyer column missing.")
    else:
        X, y, fcols, enc = prepare_classification_data(df, buyer_col)

        models = [
            (DecisionTreeClassifier(max_depth=5, random_state=42), "Decision Tree"),
            (RandomForestClassifier(n_estimators=300, random_state=42), "Random Forest"),
            (GradientBoostingClassifier(random_state=42), "Gradient Boosting"),
        ]

        results = []
        trained_models = {}

        for mdl, name in models:
            final_model, metrics = crossval_evaluate(mdl, X, y, name)
            results.append(metrics)
            trained_models[name] = final_model

        res_df = pd.DataFrame(results)
        st.dataframe(res_df.style.format("{:.3f}"))

        st.subheader("Top Customer Preferences (Real Feature Importance)")
        pref = compute_preference_importance(df)
        if pref is not None:
            top_pref = pref.head(20)
            fig = px.bar(
                x=top_pref.values,
                y=top_pref.index,
                orientation="h",
                title="Most Frequent Features / Concerns / Activities",
            )
            st.plotly_chart(fig, use_container_width=True)

        rf_model = trained_models["Random Forest"]


# =============================================================================
# 📎 ASSOCIATION RULES — SIMPLE EXPLANATION
# =============================================================================

with tabs[2]:
    st.header("📎 Association Rules — Q25, Q27, Q9 Only")

    trans_df, rules = run_association_rules(df)

    if rules is None or rules.empty:
        st.warning("No rules generated.")
    else:
        st.subheader("Top 10 Easy-to-Understand Rules")

        top10 = rules.head(10)
        for _, row in top10.iterrows():
            A = ", ".join(list(row["antecedents"]))
            C = ", ".join(list(row["consequents"]))
            st.markdown(
                f"✔ **People who selected:** *{A}*  
                ➜ **Also frequently selected:** *{C}*  
                (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})"
            )

        with st.expander("Full Rule Table"):
            st.dataframe(rules)


# =============================================================================
# 👥 CLUSTERING — PCA + 3D PLOT
# =============================================================================

with tabs[3]:

    st.header("👥 Customer Segmentation — PCA + 3D Clustering")

    cluster_vars = [
        "Q1_Age_Group", "Q2_Gender", "Q3_Location_Type", "Q4_Annual_Income",
        "Q5_Education_Level", "Q6_Employment_Status", "Q7_Occupation_Category",
        "Q8_Outdoor_Frequency", "Q10_Camping_Frequency", "Q13_Emergency_Importance",
        "Q14_Travel_Frequency", "Q16_Sustainability_Importance", "Q17_Tech_Relationship",
        "Q19_Powerbank_Ownership", "Q21_Power_Issues_Frequency"
    ]

    df_c, X_c = prepare_clustering_matrix(df, cluster_vars)

    # Elbow Method
    SSE = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_c)
        SSE.append(km.inertia_)

    fig = px.line(x=list(range(2, 9)), y=SSE, markers=True, title="Elbow Method")
    st.plotly_chart(fig, use_container_width=True)

    k_choice = st.slider("Choose Number of Clusters", 2, 8, 4)

    km = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
    df_c["Cluster"] = km.fit_predict(X_c)

    df["Cluster"] = df_c["Cluster"]

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_c)

    df_c["PC1"], df_c["PC2"], df_c["PC3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]

    fig3d = px.scatter_3d(
        df_c,
        x="PC1", y="PC2", z="PC3",
        color="Cluster",
        hover_data=["Q1_Age_Group", "Q2_Gender", "Q4_Annual_Income"],
        title="3D PCA Cluster Visualization",
    )
    st.plotly_chart(fig3d, use_container_width=True)

    summ, best = summarize_clusters(df)
    st.subheader("Cluster Summary")
    st.dataframe(summ)
    st.success(f"🎯 Best Cluster to Target: **Cluster {best}**")


# =============================================================================
# 💵 REGRESSION (PRICE)
# =============================================================================

with tabs[4]:

    st.header("💵 Regression — Price Prediction")

    if price_col is None:
        st.warning("Price column missing.")
    else:
        Xr, yr, _, _ = prepare_regression_data(df, price_col)
        Xtr, Xts, ytr, yts = train_test_split(Xr, yr, test_size=0.2, random_state=42)

        reg = LinearRegression()
        reg.fit(Xtr, ytr)
        preds = reg.predict(Xts)

        st.write(f"MAE: {mean_absolute_error(yts, preds):.2f}")
        st.write(f"RMSE: {mean_squared_error(yts, preds, squared=False):.2f}")
        st.write(f"R²: {r2_score(yts, preds):.3f}")

        rec = recommend_price(df)
        if rec:
            st.success(f"🎯 Recommended Price: **${rec}**")


# =============================================================================
# 🧪 SCORE NEW CUSTOMERS
# =============================================================================

with tabs[5]:

    st.header("🧪 Score New Customers")

    newfile = st.file_uploader("Upload new customer CSV", type=["csv"], key="new_customers")

    if newfile:
        newdf = pd.read_csv(newfile)

        fcols = [c for c in newdf.columns if c.startswith("Q") and c != "Q30_Purchase_Likelihood"]
        Xnew = pd.get_dummies(newdf[fcols], drop_first=True)

        for c in X.columns:
            if c not in Xnew.columns:
                Xnew[c] = 0
        Xnew = Xnew[X.columns]

        newdf["Predicted_Buyer"] = rf_model.predict(Xnew)
        newdf["Buyer_Probability"] = rf_model.predict_proba(Xnew)[:, 1]

        st.dataframe(newdf.head())

        st.download_button(
            "Download Predictions",
            newdf.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
