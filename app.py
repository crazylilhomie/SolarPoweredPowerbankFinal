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
    """Prepare X, y for classification using all Q-columns except Q30."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q30_Purchase_Likelihood"]
    df_model = df.dropna(subset=[target_col]).copy()

    X_raw = df_model[feature_cols]
    y = df_model[target_col].astype(int)

    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y, feature_cols, X.columns.tolist()


def prepare_regression_data(df, target_col):
    """Prepare X, y for regression using all Q-columns except Q31."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    feature_cols = [c for c in q_cols if c != "Q31_Max_Price_Willing_To_Pay"]
    df_model = df.dropna(subset=[target_col]).copy()

    X_raw = df_model[feature_cols]
    y = df_model[target_col].astype(float)

    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y, feature_cols, X.columns.tolist()


def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name):
    """Fit a classifier and compute evaluation metrics."""
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
    """
    Run Apriori association rules on selected multi-select questions.
    Multi-select answers are assumed to be separated by '|'.
    """
    transactions = []
    for _, row in df[multi_cols].iterrows():
        basket = []
        for col in multi_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip() != "":
                parts = [p.strip() for p in val.split("|") if p.strip()]
                for p in parts:
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


def prepare_clustering_matrix(df, cluster_vars):
    """Prepare data matrix for clustering (K-Means) from given variables."""
    existing = [c for c in cluster_vars if c in df.columns]
    if not existing:
        return None, None

    df_cluster = df.dropna(subset=existing).copy()
    X_cluster = pd.get_dummies(df_cluster[existing], drop_first=True)
    return df_cluster, X_cluster


def summarize_clusters(df):
    """Summarize clusters with buyer rate and average price."""
    if "Cluster" not in df.columns:
        return None, None

    summary = df.groupby("Cluster").agg(
        Count=("Cluster", "size"),
        Buyer_Rate=("Buyer", "mean"),
        Avg_Price=("Price_numeric", "mean")
    )

    # Choose cluster with highest buyer rate as best
    best_cluster = summary["Buyer_Rate"].idxmax()
    return summary, best_cluster


def recommend_price(df):
    """
    Recommend final price based on buyers' willingness to pay (Price_numeric)
    among respondents classified as Buyer = 1.
    """
    if "Buyer" not in df.columns or "Price_numeric" not in df.columns:
        return None
    buyers = df[(df["Buyer"] == 1) & df["Price_numeric"].notna()]
    if buyers.empty:
        return None
    median_price = buyers["Price_numeric"].median()
    recommended = int(5 * round(median_price / 5))  # round to nearest 5
    return recommended


def compute_preference_importance(df):
    """
    Compute 'real' feature importance as frequency of preferences in:
    Q25_Important_Features, Q28_Preferred_Capacity, Q34_Purchase_Motivators, Q35_Trusted_Brands
    """
    pref_cols = [
        "Q25_Important_Features",
        "Q28_Preferred_Capacity",
        "Q34_Purchase_Motivators",
        "Q35_Trusted_Brands"
    ]
    all_prefs = []

    for col in pref_cols:
        if col in df.columns:
            items = (
                df[col]
                .dropna()
                .astype(str)
                .str.split("|")
                .explode()
                .str.strip()
            )
            all_prefs.extend(items.tolist())

    if not all_prefs:
        return None

    pref_series = pd.Series(all_prefs).value_counts()
    return pref_series


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Solar Power Bank Launch Analytics Dashboard", layout="wide")

st.title("🔋 Solar Power Bank Launch Analytics Dashboard")

st.markdown(
    """
Upload your survey data (600+ respondents) and explore:

- **Classification** (Purchase likelihood: Buyer vs Non-buyer)
- **Association Rules** (Customer behaviour patterns)
- **Clustering** (Customer segments and target segment)
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
    "🧪 Score New Customers",
])

# -----------------------------------------------------------------------------
# Overview Tab (smaller, cleaner graphs)
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("📊 Key Survey Insights")

    st.markdown("### 👀 Quick Visual Overview")

    overview_cols = [
        "Q1_Age_Group",
        "Q2_Gender",
        "Q3_Location_Type",
        "Q8_Outdoor_Frequency",
        "Q12_Emergency_Kit",
        "Q13_Emergency_Importance",
        "Q23_Solar_Powerbank_Awareness",
        "Q28_Preferred_Capacity",
        "Q30_Purchase_Likelihood",
        "Q31_Max_Price_Willing_To_Pay",
    ]
    available_cols = [c for c in overview_cols if c in df.columns]

    col_left, col_right = st.columns(2)

    for i, col in enumerate(available_cols):
        target_col = col_left if i % 2 == 0 else col_right
        with target_col:
            st.subheader(col.replace("_", " "))
            counts = df[col].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# Classification Tab
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("🤖 Classification: Purchase Likelihood (Buyer vs Non-buyer)")

    if buyer_col is None or df[buyer_col].isna().all():
        st.warning("Buyer target could not be created from Q30_Purchase_Likelihood. Check data values.")
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
            "Decision Tree",
        )
        rf_model, rf_metrics = evaluate_classifier(
            RandomForestClassifier(n_estimators=200, random_state=42),
            X_train, X_test, y_train, y_test,
            "Random Forest",
        )
        gb_model, gb_metrics = evaluate_classifier(
            GradientBoostingClassifier(random_state=42),
            X_train, X_test, y_train, y_test,
            "Gradient Boosting",
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

        # REAL customer preference importance (from survey responses)
        st.subheader("⭐ What Features Do Customers Prefer? (Survey-based Importance)")
        pref_series = compute_preference_importance(df)
        if pref_series is not None:
            top_prefs = pref_series.head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=top_prefs.values, y=top_prefs.index, palette="viridis", ax=ax)
            ax.set_xlabel("Number of Mentions in Survey")
            ax.set_ylabel("")
            ax.set_title("Top 20 Preferred Features / Motivators / Brands")
            st.pyplot(fig)
        else:
            st.info("Could not compute preferences (check Q25, Q28, Q34, Q35).")

        # Optional: ML feature importance under expander
        with st.expander("🔍 Model Feature Importance (Random Forest – technical view)"):
            importances = rf_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": importances,
            }).sort_values(by="Importance", ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
            ax.set_title("Top 20 Features Driving the Model")
            st.pyplot(fig)

        st.markdown(
            "The **Random Forest** model above will also be used for scoring new customers "
            "in the **Score New Customers** tab."
        )

# -----------------------------------------------------------------------------
# Association Rules Tab – Simplified Insights
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("📎 Simple Customer Pattern Insights (Association Rules)")

    multi_cols = [
        "Q9_Outdoor_Activities",
        "Q11_Camping_Type",
        "Q18_Portable_Devices",
        "Q22_Charging_Challenges",
        "Q25_Important_Features",
        "Q27_Concerns",
        "Q29_Purchase_Channels",
        "Q32_Bundle_Preferences",
        "Q34_Purchase_Motivators",
        "Q35_Trusted_Brands",
    ]
    multi_cols = [c for c in multi_cols if c in df.columns]

    if not multi_cols:
        st.warning("No multi-select questions found in data for association rules.")
    else:
        st.markdown(f"Using multi-select questions: {', '.join(multi_cols)}")

        min_support = st.slider("Minimum support", 0.01, 0.2, 0.05, step=0.01)
        min_lift = st.slider("Minimum lift", 0.8, 3.0, 1.0, step=0.1)

        trans_df, rules = run_association_rules(df, multi_cols, min_support=min_support, metric="lift", min_threshold=min_lift)

        if rules is None or rules.empty:
            st.warning("No association rules found with current thresholds. Try lowering minimum support/lift.")
        else:
            st.subheader("🧠 Top 10 Easy-to-Understand Insights")

            simple_rules = []
            for _, row in rules.head(10).iterrows():
                A = ", ".join(list(row["antecedents"]))
                C = ", ".join(list(row["consequents"]))
                conf = round(row["confidence"] * 100, 1)
                lift = round(row["lift"], 2)
                text = (
                    f"**People who chose _{A}_ also often chose _{C}_** "
                    f"(confidence: {conf}%, lift: {lift})"
                )
                simple_rules.append(text)

            for s in simple_rules:
                st.markdown(f"✔️ {s}")

            with st.expander("📋 See raw rule table (advanced)"):
                display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
                rules_display = rules[display_cols].copy()
                rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(list(x)))
                st.dataframe(rules_display)

# -----------------------------------------------------------------------------
# Clustering Tab – Elbow + Simple Demographics
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("👥 Customer Segmentation (K-Means)")

    cluster_vars = [
        "Q1_Age_Group",
        "Q2_Gender",
        "Q3_Location_Type",
        "Q4_Annual_Income",
        "Q5_Education_Level",
        "Q6_Employment_Status",
        "Q7_Occupation_Category",
        "Q8_Outdoor_Frequency",
        "Q10_Camping_Frequency",
        "Q13_Emergency_Importance",
        "Q14_Travel_Frequency",
        "Q16_Sustainability_Importance",
        "Q17_Tech_Relationship",
        "Q19_Powerbank_Ownership",
        "Q21_Power_Issues_Frequency",
    ]

    df_cluster, X_cluster = prepare_clustering_matrix(df, cluster_vars)

    if df_cluster is None:
        st.warning("Clustering variables not found or insufficient data.")
    else:
        st.subheader("📉 Elbow Method – Choose Number of Clusters")

        sse = []
        K_range = range(2, 9)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_cluster)
            sse.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_range, sse, marker="o")
        ax.set_xlabel("Number of clusters (K)")
        ax.set_ylabel("SSE (Within-cluster sum of squares)")
        ax.set_title("Elbow Plot for K-Means")
        st.pyplot(fig)

        k_choice = st.slider("Select number of clusters for segmentation", 2, 8, 4)

        kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
        df_cluster["Cluster"] = kmeans.fit_predict(X_cluster)

        # Merge cluster labels back into main df by index
        df = df.copy()
        df["Cluster"] = df_cluster["Cluster"]

        st.subheader("Cluster Performance Summary")
        cluster_summary, best_cluster = summarize_clusters(df)
        st.dataframe(cluster_summary)

        st.markdown(f"### 🎯 Recommended Early-Target Cluster: **Cluster {best_cluster}**")

        st.markdown(
            "This cluster has the **highest buyer rate** (proportion of Buyer = 1), "
            "so it is statistically the easiest early-win target segment."
        )

        st.subheader("📌 Cluster Demographics (Most Common Values)")
        demo_cols = ["Q1_Age_Group", "Q2_Gender", "Q3_Location_Type"]
        existing_demo = [c for c in demo_cols if c in df.columns]

        if existing_demo:
            demo_summary = df.groupby("Cluster")[existing_demo].agg(
                lambda x: x.value_counts().index[0] if not x.value_counts().empty else None
            )
            st.dataframe(demo_summary)
        else:
            st.info("No demographic columns found to summarize.")

        st.subheader("Cluster Size Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df["Cluster"], ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Regression Tab
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("💵 Price Prediction (Linear Regression on Q31_Max_Price_Willing_To_Pay)")

    if price_col is None or df[price_col].isna().all():
        st.warning("Price_numeric could not be created from Q31_Max_Price_Willing_To_Pay. Check data values.")
    else:
        X_reg, y_reg, reg_features, reg_encoded_cols = prepare_regression_data(df, price_col)
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        reg_model = LinearRegression()
        reg_model.fit(X_train_r, y_train_r)

        y_pred_r = reg_model.predict(X_test_r)

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
Upload a CSV with the same question columns (Q1_Age_Group to Q39_Disaster_Experience)
for new or expected customers.

The app will use the trained **Random Forest** classifier to label each row as:

- `1` → Buyer  
- `0` → Non-buyer
"""
        )

        new_file = st.file_uploader("Upload new customer CSV", type=["csv"], key="new_customers")

        if new_file is not None:
            new_df = pd.read_csv(new_file)

            q_cols_new = [c for c in new_df.columns if c.startswith("Q")]
            # Exclude purchase likelihood from features if accidentally present
            feature_cols_new = [c for c in q_cols_new if c != "Q30_Purchase_Likelihood"]

            X_new_raw = new_df[feature_cols_new]
            X_new_enc = pd.get_dummies(X_new_raw, drop_first=True)

            # Align columns with training data from classification
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
                mime="text/csv",
            )
