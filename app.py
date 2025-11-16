import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def create_buyer_flag(df):
    """Create binary Buyer target from Q30_Purchase_Likelihood."""
    col = "Q30_Purchase_Likelihood"
    if col not in df.columns:
        st.warning(f"{col} column not found in data. Classification may not run correctly.")
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
    """Convert Q31_Max_Price_Willing_To_Pay bands to numeric midpoints."""
    col = "Q31_Max_Price_Willing_To_Pay"
    if col not in df.columns:
        st.warning(f"{col} column not found in data. Regression may not run correctly.")
        return df, None

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
    """Prepare X, y for classification."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q30_Purchase_Likelihood"]
    df_model = df.dropna(subset=[target_col]).copy()

    X_raw = df_model[feature_cols]
    y = df_model[target_col].astype(int)

    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y, feature_cols, X.columns.tolist()


def prepare_regression_data(df, target_col):
    """Prepare X, y for regression."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q31_Max_Price_Willing_To_Pay"]
    df_model = df.dropna(subset=[target_col]).copy()

    X_raw = df_model[feature_cols]
    y = df_model[target_col].astype(float)

    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y, feature_cols, X.columns.tolist()


def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_scores = model.decision_function(X_test)
        roc = roc_auc_score(y_test, y_scores)

    return model, {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc,
    }


def run_association_rules(df, multi_cols, min_support=0.05, metric="lift", min_threshold=1.0):
    """Run Apriori on multi-select questions separated by |."""
    transactions = []
    for _, row in df[multi_cols].iterrows():
        basket = []
        for col in multi_cols:
            val = row[col]
            if isinstance(val, str):
                parts = [p.strip() for p in val.split("|") if p.strip()]
                for p in parts:
                    basket.append(f"{col}: {p}")
        if basket:
            transactions.append(basket)

    if len(transactions) == 0:
        return None, None

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent = apriori(trans_df, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return trans_df, None

    rules = association_rules(frequent, metric=metric, min_threshold=min_threshold)
    if rules.empty:
        return trans_df, None

    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)
    return trans_df, rules


def run_clustering(df):
    cluster_vars = [
        "Q1_Age_Group", "Q2_Gender", "Q3_Location_Type", "Q4_Annual_Income",
        "Q5_Education_Level", "Q6_Employment_Status", "Q7_Occupation_Category",
        "Q8_Outdoor_Frequency", "Q10_Camping_Frequency", "Q13_Emergency_Importance",
        "Q14_Travel_Frequency", "Q16_Sustainability_Importance", "Q17_Tech_Relationship",
        "Q19_Powerbank_Ownership", "Q21_Power_Issues_Frequency"
    ]
    existing = [c for c in cluster_vars if c in df.columns]

    df_cluster = df.dropna(subset=existing).copy()
    X_cluster = pd.get_dummies(df_cluster[existing], drop_first=True)

    n_clusters = st.sidebar.slider("Number of clusters (K)", 3, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_cluster)

    df = df.copy()
    df["Cluster"] = df_cluster["Cluster"]

    return df, df_cluster, kmeans


def summarize_clusters(df):
    if "Cluster" not in df.columns:
        return None, None

    summary = df.groupby("Cluster").agg(
        Count=("Cluster", "size"),
        Buyer_Rate=("Buyer", "mean"),
        Avg_Price=("Price_numeric", "mean")
    )

    best_cluster = summary["Buyer_Rate"].idxmax()
    return summary, best_cluster


def recommend_price(df):
    buyers = df[(df["Buyer"] == 1) & df["Price_numeric"].notna()]
    if buyers.empty:
        return None
    median_price = buyers["Price_numeric"].median()
    return int(5 * round(median_price / 5))


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Solar Power Bank Dashboard", layout="wide")
st.title("🔋 Solar Power Bank Launch Analytics Dashboard")

uploaded = st.file_uploader("Upload your survey CSV", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Data Loaded: {df.shape[0]} rows")

df, buyer_col = create_buyer_flag(df)
df, price_col = create_price_numeric(df)

tabs = st.tabs(["📊 Overview", "🤖 Classification", "📎 Association Rules",
                "👥 Clustering", "💵 Regression", "🧪 Score Customers"])

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("📊 Overview")
    overview_cols = [
        "Q1_Age_Group", "Q2_Gender", "Q3_Location_Type", "Q8_Outdoor_Frequency",
        "Q12_Emergency_Kit", "Q13_Emergency_Importance", "Q23_Solar_Powerbank_Awareness",
        "Q28_Preferred_Capacity", "Q30_Purchase_Likelihood", "Q31_Max_Price_Willing_To_Pay"
    ]
    for col in overview_cols:
        if col in df.columns:
            st.subheader(col)
            fig, ax = plt.subplots()
            sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("🤖 Classification – Buyer Prediction")

    X, y, _, _ = prepare_classification_data(df, "Buyer")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = [
        (DecisionTreeClassifier(max_depth=5, random_state=42), "Decision Tree"),
        (RandomForestClassifier(n_estimators=200, random_state=42), "Random Forest"),
        (GradientBoostingClassifier(random_state=42), "Gradient Boosting")
    ]

    results = []
    for model, name in models:
        m, metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        if name == "Random Forest":
            rf_model = m

    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(results))

    st.subheader("Feature Importance (Random Forest)")
    fi = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# Association Rules
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("📎 Association Rule Mining")

    multi_cols = [
        "Q9_Outdoor_Activities", "Q11_Camping_Type", "Q18_Portable_Devices",
        "Q22_Charging_Challenges", "Q25_Important_Features", "Q27_Concerns",
        "Q29_Purchase_Channels", "Q32_Bundle_Preferences",
        "Q34_Purchase_Motivators", "Q35_Trusted_Brands"
    ]
    multi_cols = [c for c in multi_cols if c in df.columns]

    min_support = st.slider("Minimum support", 0.01, 0.2, 0.05)
    min_lift = st.slider("Minimum lift", 0.5, 3.0, 1.0)

    _, rules = run_association_rules(df, multi_cols, min_support, "lift", min_lift)

    if rules is not None:
        st.subheader("Top 10 Association Rules")
        top10 = rules.head(10).copy()
        top10["antecedents"] = top10["antecedents"].apply(lambda x: ", ".join(list(x)))
        top10["consequents"] = top10["consequents"].apply(lambda x: ", ".join(list(x)))
        st.dataframe(top10)
    else:
        st.warning("No rules found at selected thresholds.")

# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("👥 Customer Segmentation")

    df_clustered, df_cluster_only, kmeans = run_clustering(df)
    summary, best_cluster = summarize_clusters(df_clustered)

    st.subheader("Cluster Summary")
    st.dataframe(summary)

    st.markdown(f"### 🎯 Best Target Cluster: **Cluster {best_cluster}**")

    fig, ax = plt.subplots()
    sns.countplot(x=df_clustered["Cluster"], ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("💵 Price Prediction")

    Xr, yr, _, _ = prepare_regression_data(df, "Price_numeric")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    reg = LinearRegression().fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    st.write(f"MAE: {mean_absolute_error(y_test_r, y_pred_r):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.2f}")
    st.write(f"R²: {r2_score(y_test_r, y_pred_r):.3f}")

    rec_price = recommend_price(df)
    if rec_price:
        st.subheader(f"🎯 Recommended Launch Price: **${rec_price}**")

# -----------------------------------------------------------------------------
# Score New Customers
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("🧪 Score New Customers")

    new = st.file_uploader("Upload new customer CSV", type=["csv"], key="score_csv")

    if new:
        new_df = pd.read_csv(new)

        # Same preprocessing as training
        q_cols = [c for c in new_df.columns if c.startswith("Q")]
        feat_cols = [c for c in q_cols if c != "Q30_Purchase_Likelihood"]

        X_new = pd.get_dummies(new_df[feat_cols], drop_first=True)
        missing = set(X.columns) - set(X_new.columns)
        for c in missing:
            X_new[c] = 0
        X_new = X_new[X.columns]

        new_df["Predicted_Buyer"] = rf_model.predict(X_new)
        new_df["Probability"] = rf_model.predict_proba(X_new)[:, 1]

        st.dataframe(new_df.head())

        st.download_button(
            "Download predictions",
            new_df.to_csv(index=False),
            "scored_customers.csv",
            "text/csv"
        )
