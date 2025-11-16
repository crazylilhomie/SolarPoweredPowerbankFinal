import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
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

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

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


def crossval_evaluate(model, X, y, model_name, n_splits=5):
    """5-fold stratified cross-validation evaluation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_all = []
    y_score_all = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        mdl = clone(model)
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        y_true_all.append(y_te)
        y_pred_all.append(y_pred)

        if hasattr(mdl, "predict_proba"):
            y_score_all.append(mdl.predict_proba(X_te)[:, 1])
        else:
            y_score_all.append(mdl.decision_function(X_te))

    y_true_cat = pd.concat(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    y_score_cat = np.concatenate(y_score_all)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true_cat, y_pred_cat),
        "Precision": precision_score(y_true_cat, y_pred_cat, zero_division=0),
        "Recall": recall_score(y_true_cat, y_pred_cat, zero_division=0),
        "F1-score": f1_score(y_true_cat, y_pred_cat, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true_cat, y_score_cat),
    }

    # Fit final model on full data for later scoring
    final_model = clone(model)
    final_model.fit(X, y)

    return final_model, metrics


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

- **Classification** (Purchase likelihood: Buyer vs Non-buyer, with 5-fold CV)
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
# OVERVIEW TAB – 10 GRAPHS FROM REPORT
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("📊 Descriptive Analytics – Key Graphs")

    # Graph 1: Key Demographics (Gender & Location) – Pie Charts
    st.subheader("Graph 1: Key Demographics (Gender & Location Type)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if "Q2_Gender" in df.columns:
        gender_counts = df["Q2_Gender"].value_counts()
        axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%", startangle=90)
        axes[0].set_title("Gender Distribution")
    else:
        axes[0].text(0.5, 0.5, "Q2_Gender not found", ha="center")

    if "Q3_Location_Type" in df.columns:
        loc_counts = df["Q3_Location_Type"].value_counts()
        axes[1].pie(loc_counts.values, labels=loc_counts.index, autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Location Type Distribution")
    else:
        axes[1].text(0.5, 0.5, "Q3_Location_Type not found", ha="center")

    st.pyplot(fig)

    # Graph 2: Heatmap - Purchase Likelihood vs Age Group
    st.subheader("Graph 2: Heatmap – Purchase Likelihood vs Age Group")
    if "Q1_Age_Group" in df.columns and "Q30_Purchase_Likelihood" in df.columns:
        ct = pd.crosstab(df["Q1_Age_Group"], df["Q30_Purchase_Likelihood"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_xlabel("Purchase Likelihood")
        ax.set_ylabel("Age Group")
        st.pyplot(fig)
    else:
        st.info("Required columns Q1_Age_Group and/or Q30_Purchase_Likelihood not found.")

    # Graph 3: Target Income Level Distribution – Vertical Bar
    st.subheader("Graph 3: Target Income Level Distribution")
    if "Q4_Annual_Income" in df.columns:
        income_counts = df["Q4_Annual_Income"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=income_counts.index, y=income_counts.values, ax=ax, palette="viridis")
        ax.set_xlabel("Annual Income Bracket")
        ax.set_ylabel("Number of Respondents")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("Column Q4_Annual_Income not found.")

    # Graph 4: Preferred Power Bank Capacity – Vertical Bar
    st.subheader("Graph 4: Preferred Power Bank Capacity")
    if "Q28_Preferred_Capacity" in df.columns:
        cap_counts = df["Q28_Preferred_Capacity"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=cap_counts.index, y=cap_counts.values, ax=ax, palette="viridis")
        ax.set_xlabel("Preferred Capacity (mAh range)")
        ax.set_ylabel("Number of Respondents")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("Column Q28_Preferred_Capacity not found.")

    # Graph 5: Preferred Purchase Channels – Horizontal Bar + Top 3
    st.subheader("Graph 5: Preferred Purchase Channels")
    if "Q29_Purchase_Channels" in df.columns:
        temp = df["Q29_Purchase_Channels"].dropna().astype(str).str.split("|").explode().str.strip()
        channel_counts = temp.value_counts()
        total = channel_counts.sum()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x=channel_counts.values, y=channel_counts.index, ax=ax, palette="viridis")
        ax.set_xlabel("Number of Mentions")
        ax.set_ylabel("Purchase Channel")
        st.pyplot(fig)

        # Medal podium (Top 3)
        st.markdown("#### 🥇 Top 3 Purchase Channels")
        for i, (ch, cnt) in enumerate(channel_counts.head(3).items(), start=1):
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
            pct = (cnt / total) * 100 if total > 0 else 0
            st.markdown(f"{medal} **{ch}** – {cnt} mentions ({pct:.1f}%)")
    else:
        st.info("Column Q29_Purchase_Channels not found.")

    # Graph 6: Most Popular Bundle Preferences – Horizontal Bar + Top 3
    st.subheader("Graph 6: Most Popular Bundle Preferences")
    if "Q32_Bundle_Preferences" in df.columns:
        temp = df["Q32_Bundle_Preferences"].dropna().astype(str).str.split("|").explode().str.strip()
        bundle_counts = temp.value_counts()
        total_b = bundle_counts.sum()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x=bundle_counts.values, y=bundle_counts.index, ax=ax, palette="viridis")
        ax.set_xlabel("Number of Mentions")
        ax.set_ylabel("Bundle Type")
        st.pyplot(fig)

        st.markdown("#### 🥇 Top 3 Bundles")
        for i, (bndl, cnt) in enumerate(bundle_counts.head(3).items(), start=1):
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
            pct = (cnt / total_b) * 100 if total_b > 0 else 0
            st.markdown(f"{medal} **{bndl}** – {cnt} mentions ({pct:.1f}%)")
    else:
        st.info("Column Q32_Bundle_Preferences not found.")

    # Graph 7: Heatmap – Top Features vs Primary Use Case
    st.subheader("Graph 7: Heatmap – Top Features vs Primary Use Case")
    if "Q25_Important_Features" in df.columns and "Q26_Primary_Use_Case" in df.columns:
        df_feat = df[["Q25_Important_Features", "Q26_Primary_Use_Case"]].dropna()
        all_features = (
            df_feat["Q25_Important_Features"]
            .astype(str)
            .str.split("|")
            .explode()
            .str.strip()
        )
        top5_features = all_features.value_counts().head(5).index

        use_cases = df_feat["Q26_Primary_Use_Case"].dropna().unique()
        heat_data = pd.DataFrame(0, index=top5_features, columns=use_cases)

        for _, row in df_feat.iterrows():
            feats = [f.strip() for f in str(row["Q25_Important_Features"]).split("|") if f.strip()]
            uc = row["Q26_Primary_Use_Case"]
            for f in feats:
                if f in top5_features:
                    heat_data.loc[f, uc] += 1

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(heat_data, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
        ax.set_xlabel("Primary Use Case")
        ax.set_ylabel("Top Features")
        st.pyplot(fig)
    else:
        st.info("Columns Q25_Important_Features and/or Q26_Primary_Use_Case not found.")

    # Graph 8: Employment Status vs Price Willingness – Boxplot
    st.subheader("Graph 8: Employment Status vs Willingness to Pay (Boxplot)")
    if "Q6_Employment_Status" in df.columns and "Price_numeric" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Q6_Employment_Status", y="Price_numeric", data=df, ax=ax)
        ax.set_xlabel("Employment Status")
        ax.set_ylabel("Max Price Willing to Pay (USD)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("Columns Q6_Employment_Status and/or Price_numeric not found.")

    # Graph 9: Household Size vs Bundle Preference – Grouped Bar Chart
    st.subheader("Graph 9: Household Size vs Top Bundle Preferences")
    if "Q36_Household_Size" in df.columns and "Q32_Bundle_Preferences" in df.columns:
        temp = df[["Q36_Household_Size", "Q32_Bundle_Preferences"]].dropna()
        temp = temp.assign(Bundle=temp["Q32_Bundle_Preferences"].astype(str).str.split("|")).explode("Bundle")
        temp["Bundle"] = temp["Bundle"].str.strip()
        top5_bundles = temp["Bundle"].value_counts().head(5).index
        temp = temp[temp["Bundle"].isin(top5_bundles)]

        pivot = pd.pivot_table(
            temp,
            index="Q36_Household_Size",
            columns="Bundle",
            aggfunc="size",
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Household Size")
        ax.set_ylabel("Number of Mentions")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    else:
        st.info("Columns Q36_Household_Size and/or Q32_Bundle_Preferences not found.")

    # Graph 10: Gender vs Feature Preferences – Butterfly Chart
    st.subheader("Graph 10: Gender vs Feature Preferences (Butterfly Chart)")
    if "Q2_Gender" in df.columns and "Q25_Important_Features" in df.columns:
        temp = df[["Q2_Gender", "Q25_Important_Features"]].dropna()
        temp = temp.assign(Feature=temp["Q25_Important_Features"].astype(str).str.split("|")).explode("Feature")
        temp["Feature"] = temp["Feature"].str.strip()

        # focus on Male/Female
        temp = temp[temp["Q2_Gender"].isin(["Male", "Female"])]
        overall_top = temp["Feature"].value_counts().head(10).index

        gender_feat = (
            temp[temp["Feature"].isin(overall_top)]
            .groupby(["Feature", "Q2_Gender"])
            .size()
            .unstack(fill_value=0)
        )

        # Ensure both columns exist
        for g in ["Male", "Female"]:
            if g not in gender_feat.columns:
                gender_feat[g] = 0

        gender_feat = gender_feat.loc[overall_top]  # keep order

        male_counts = gender_feat["Male"].values
        female_counts = gender_feat["Female"].values

        y = np.arange(len(overall_top))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(y, male_counts, color="#1f77b4", label="Male")
        ax.barh(y, -female_counts, color="#ff7f0e", label="Female")
        ax.set_yticks(y)
        ax.set_yticklabels(overall_top)
        ax.axvline(0, color="black")
        ax.set_xlabel("Mentions (Male right / Female left)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Columns Q2_Gender and/or Q25_Important_Features not found.")

# -----------------------------------------------------------------------------
# Classification Tab – 5-FOLD CROSS-VALIDATION
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("🤖 Classification: Purchase Likelihood (Buyer vs Non-buyer with 5-fold CV)")

    if buyer_col is None or df[buyer_col].isna().all():
        st.warning("Buyer target could not be created from Q30_Purchase_Likelihood. Check data values.")
    else:
        X_cls, y_cls, cls_features, cls_encoded_cols = prepare_classification_data(df, buyer_col)

        # 5-fold CV for each model
        dt_model, dt_metrics = crossval_evaluate(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            X_cls, y_cls,
            "Decision Tree (5-fold CV)"
        )
        rf_model, rf_metrics = crossval_evaluate(
            RandomForestClassifier(n_estimators=200, random_state=42),
            X_cls, y_cls,
            "Random Forest (5-fold CV)"
        )
        gb_model, gb_metrics = crossval_evaluate(
            GradientBoostingClassifier(random_state=42),
            X_cls, y_cls,
            "Gradient Boosting (5-fold CV)"
        )

        metrics_df = pd.DataFrame([dt_metrics, rf_metrics, gb_metrics])
        st.subheader("📋 Cross-Validated Model Performance (5 folds)")
        st.dataframe(metrics_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }))

        # Real customer preference importance from survey responses
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

        # Optional technical ML feature importance
        with st.expander("🔍 Model Feature Importance (Random Forest – technical view)"):
            importances = rf_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": X_cls.columns,
                "Importance": importances,
            }).sort_values(by="Importance", ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
            ax.set_title("Top 20 Features Driving the Model")
            st.pyplot(fig)

        st.markdown(
            "The **Random Forest** model above (trained with 5-fold CV and then on full data) "
            "will be used for scoring new customers in the **Score New Customers** tab."
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
