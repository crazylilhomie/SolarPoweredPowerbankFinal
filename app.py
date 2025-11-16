import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
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
    """Create binary target from Q30."""
    if "Q30" not in df.columns:
        st.warning("Q30 column not found in data. Classification may not run correctly.")
        return df, None

    buyer_map = {
        "Definitely will purchase (90-100% likely)": 1,
        "Very likely to purchase (70-89%)": 1,
        "Moderately likely (50-69%)": 0,
        "Somewhat likely (30-49%)": 0,
        "Unlikely (10-29%)": 0,
        "Definitely will NOT purchase (0-9%)": 0,
    }
    df["Buyer"] = df["Q30"].map(buyer_map)
    return df, "Buyer"


def create_price_numeric(df):
    """Convert Q31 price bands to numeric midpoints."""
    if "Q31" not in df.columns:
        st.warning("Q31 column not found in data. Regression may not run correctly.")
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
    df["Price_numeric"] = df["Q31"].map(price_map)
    return df, "Price_numeric"


def prepare_classification_data(df, target_col):
    """Prepare X, y for classification using all Q-columns except Q30."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q30"]  # all Qs except target question
    df_model = df.dropna(subset=[target_col]).copy()

    X_raw = df_model[feature_cols]
    y = df_model[target_col].astype(int)

    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y, feature_cols, X.columns.tolist()


def prepare_regression_data(df, target_col):
    """Prepare X, y for regression using all Q-columns except Q31."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q31"]  # all Qs except target question
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

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc,
    }
    return model, metrics


def run_association_rules(df, multi_cols, min_support=0.05, metric="lift", min_threshold=1.0):
    """Run Apriori association rules on selected multi-select questions."""
    # Build transactions list
    transactions = []
    for _, row in df[multi_cols].iterrows():
        basket = []
        for col in multi_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str):
                # Assume multi-select values are separated by ';'
                parts = [p.strip() for p in val.split(";") if p.strip()]
                for p in parts:
                    if p.lower() in ["none of the above", "no bundle - prefer standalone product only"]:
                        continue
                    basket.append(f"{col}: {p}")
        if basket:
            transactions.append(basket)

    if len(transactions) == 0:
        return None, None

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(trans_df, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return trans_df, None

    rules = association_rules(frequent, metric=metric, min_threshold=min_threshold)
    if rules.empty:
        return trans_df, None

    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)
    return trans_df, rules


def run_clustering(df):
    """Run K-Means clustering on selected variables."""
    cluster_vars = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
                    "Q8", "Q10", "Q13", "Q14", "Q16", "Q17",
                    "Q19", "Q21"]
    existing = [c for c in cluster_vars if c in df.columns]
    if not existing:
        st.warning("No clustering variables found in data.")
        return df, None, None

    df_cluster = df.dropna(subset=existing).copy()
    X_cluster = pd.get_dummies(df_cluster[existing], drop_first=True)

    n_clusters = st.sidebar.slider("Number of clusters (K)", 3, 8, 4)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_cluster)

    # Bring cluster labels back to main df (may have NaNs for dropped rows)
    df = df.copy()
    df = df.merge(df_cluster[["Cluster"]], left_index=True, right_index=True, how="left")

    return df, df_cluster, kmeans


def summarize_clusters(df):
    if "Cluster" not in df.columns:
        return None, None

    if "Buyer" in df.columns:
        summary = df.groupby("Cluster").agg(
            Count=("Cluster", "size"),
            Buyer_Rate=("Buyer", "mean"),
            Avg_Price=("Price_numeric", "mean")
        )
    else:
        summary = df.groupby("Cluster").agg(
            Count=("Cluster", "size"),
            Avg_Price=("Price_numeric", "mean")
        )
    summary = summary.sort_index()
    if "Buyer_Rate" in summary.columns:
        best_cluster = summary["Buyer_Rate"].idxmax()
    else:
        best_cluster = summary["Avg_Price"].idxmax()
    return summary, best_cluster


def recommend_price(df):
    """Recommend final price based on buyers' willingness to pay."""
    if "Buyer" not in df.columns or "Price_numeric" not in df.columns:
        return None
    buyers = df[(df["Buyer"] == 1) & df["Price_numeric"].notna()]
    if buyers.empty:
        return None
    median_price = buyers["Price_numeric"].median()
    # Round to nearest 5
    recommended = int(5 * round(median_price / 5))
    return recommended


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Solar Power Bank Launch Analytics Dashboard", layout="wide")

st.title("🔋 Solar Power Bank Launch Analytics Dashboard")

st.markdown(
    """
Upload your survey data (600+ respondents) and explore:
- **Classification** (Purchase likelihood)
- **Association Rules** (Customer patterns)
- **Clustering** (Customer segments)
- **Regression** (Price prediction)
- **Scenario Scoring** (Label new customers)
"""
)

uploaded_file = st.file_uploader("Upload your survey CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload the survey CSV file to begin.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
st.success(f"Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Create target variables
df, buyer_col = create_buyer_flag(df)
df, price_col = create_price_numeric(df)

# Tabs
tabs = st.tabs([
    "📊 Overview",
    "🤖 Classification",
    "📎 Association Rules",
    "👥 Clustering",
    "💵 Price Regression",
    "🧪 Score New Customers"
])

# -----------------------------------------------------------------------------
# Overview Tab
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("📊 Key Survey Insights")
    st.markdown("These graphs help understand your market before modelling.")

    # 10 simple, business-useful graphs (or as many as available)
    overview_cols = [
        "Q1", "Q2", "Q3", "Q8", "Q12", "Q13",
        "Q23", "Q28", "Q30", "Q31"
    ]
    available_cols = [c for c in overview_cols if c in df.columns]

    for col in available_cols:
        st.subheader(col)
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Classification Tab
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("🤖 Classification: Purchase Likelihood (Q30 → Buyer/Non-buyer)")

    if buyer_col is None or df[buyer_col].isna().all():
        st.warning("Buyer target could not be created from Q30. Check data values.")
    else:
        X_cls, y_cls, cls_features, cls_encoded_cols = prepare_classification_data(df, buyer_col)

        test_size = st.slider("Test size (for train/test split)", 0.1, 0.4, 0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X_cls, y_cls, test_size=test_size, random_state=42, stratify=y_cls
        )

        # Train models
        dt_model, dt_metrics = evaluate_classifier(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            X_train, X_test, y_train, y_test,
            "Decision Tree"
        )
        rf_model, rf_metrics = evaluate_classifier(
            RandomForestClassifier(n_estimators=200, random_state=42),
            X_train, X_test, y_train, y_test,
            "Random Forest"
        )
        gb_model, gb_metrics = evaluate_classifier(
            GradientBoostingClassifier(random_state=42),
            X_train, X_test, y_train, y_test,
            "Gradient Boosting"
        )

        metrics_df = pd.DataFrame([dt_metrics, rf_metrics, gb_metrics])
        st.subheader("📋 Model Performance Comparison")
        st.dataframe(metrics_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }))

        # Feature importance from Random Forest
        st.subheader("⭐ Feature Importance (Random Forest)")
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 20 Important Features")
        st.pyplot(fig)

        st.markdown("Random Forest model will also be used for scoring new customers in the **Score New Customers** tab.")

# -----------------------------------------------------------------------------
# Association Rules Tab
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("📎 Association Rule Mining (Apriori)")

    multi_cols = [q for q in ["Q9", "Q11", "Q18", "Q22", "Q25", "Q27", "Q29", "Q32", "Q34", "Q35"] if q in df.columns]

    if not multi_cols:
        st.warning("No multi-select questions (Q9, Q11, Q18, Q22, Q25, Q27, Q29, Q32, Q34, Q35) found in data.")
    else:
        st.markdown(f"Using multi-select questions: {', '.join(multi_cols)}")

        min_support = st.slider("Minimum support", 0.01, 0.2, 0.05, step=0.01)
        min_lift = st.slider("Minimum lift", 0.5, 3.0, 1.0, step=0.1)

        trans_df, rules = run_association_rules(df, multi_cols, min_support=min_support, metric="lift", min_threshold=min_lift)

        if rules is None or rules.empty:
            st.warning("No association rules found with current thresholds. Try lowering minimum support/lift.")
        else:
            st.subheader("🏆 Top 10 Association Rules (Sorted by Lift)")
            top_rules = rules.head(10).copy()
            display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
            top_rules_display = top_rules[display_cols].copy()
            top_rules_display["antecedents"] = top_rules_display["antecedents"].apply(lambda x: ", ".join(list(x)))
            top_rules_display["consequents"] = top_rules_display["consequents"].apply(lambda x: ", ".join(list(x)))
            st.dataframe(top_rules_display)

# -----------------------------------------------------------------------------
# Clustering Tab
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("👥 Customer Segmentation (K-Means)")

    df_clustered, df_cluster_only, kmeans = run_clustering(df)

    if df_cluster_only is None:
        st.warning("Clustering could not be performed. Check selected variables.")
    else:
        st.subheader("Cluster Summary")
        cluster_summary, best_cluster = summarize_clusters(df_clustered)
        st.dataframe(cluster_summary)

        if best_cluster is not None:
            st.markdown(f"### 🎯 Recommended Early-Target Cluster: **Cluster {best_cluster}**")
            st.markdown(
                "This cluster has the highest **buyer rate** (or highest average willingness to pay if buyer info is missing), "
                "making it a strong candidate for early, easy success."
            )

        st.subheader("Cluster Size Distribution")
        fig, ax = plt.subplots()
        cluster_counts = df_clustered["Cluster"].value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Regression Tab
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("💵 Price Prediction (Linear Regression on Q31)")

    if price_col is None or df[price_col].isna().all():
        st.warning("Price_numeric could not be created from Q31. Check data values.")
    else:
        X_reg, y_reg, reg_features, reg_encoded_cols = prepare_regression_data(df, price_col)
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        reg_model = LinearRegression()
        reg_model.fit(X_train_r, y_train_r)

        y_pred_r = reg_model.predict(X_test_r)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test_r, y_pred_r)
        mse = mean_squared_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_r, y_pred_r)

        st.subheader("📋 Regression Performance")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R²:** {r2:.3f}")

        recommended = recommend_price(df)
        if recommended is not None:
            st.subheader("🎯 Recommended Product Price")
            st.write(
                f"Based on the **median willingness to pay** of respondents who are likely buyers, "
                f"a suggested launch price is **${recommended}**."
            )
        else:
            st.info("Could not compute recommended price (no buyer or price info).")

# -----------------------------------------------------------------------------
# Score New Customers Tab
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("🧪 Score New Customers (Will They Buy?)")

    if buyer_col is None or df[buyer_col].isna().all():
        st.warning("Classification models are not available because Buyer target could not be created.")
    else:
        st.markdown(
            """
Upload a CSV with the same question columns (Q1–Q29, Q31–Q39) **without Q30** for new or expected customers.
The app will use the trained Random Forest classifier to label each row as Buyer (1) or Non-buyer (0).
"""
        )

        new_file = st.file_uploader("Upload new customer CSV", type=["csv"], key="new_customers")

        if new_file is not None:
            new_df = pd.read_csv(new_file)

            # Use same feature columns as classification
            q_cols_new = [c for c in new_df.columns if c.startswith("Q")]
            # Ensure we exclude Q30 if present accidentally
            feature_cols_new = [c for c in q_cols_new if c != "Q30"]

            X_new_raw = new_df[feature_cols_new]
            X_new_enc = pd.get_dummies(X_new_raw, drop_first=True)

            # Align columns with training data
            missing_cols = set(X_cls.columns) - set(X_new_enc.columns)
            for c in missing_cols:
                X_new_enc[c] = 0
            X_new_enc = X_new_enc[X_cls.columns]

            new_df["Predicted_Buyer_RF"] = rf_model.predict(X_new_enc)
            if hasattr(rf_model, "predict_proba"):
                new_df["Probability_RF"] = rf_model.predict_proba(X_new_enc)[:, 1]

            st.subheader("Preview of Scored Customers")
            st.dataframe(new_df.head())

            csv_data = new_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Customers as CSV",
                data=csv_data,
                file_name="scored_customers_with_predictions.csv",
                mime="text/csv"
            )
