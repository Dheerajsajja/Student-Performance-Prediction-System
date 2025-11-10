# data_analysis.py
import matplotlib
# Use non-interactive backend (avoid macOS / thread issues)
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import math
from typing import Optional, Dict, Any, List, Tuple

sns.set(style="whitegrid")

def _fig_to_base64(fig, dpi=150):
    """Convert a matplotlib figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def _df_to_html(df: pd.DataFrame, max_rows:int=50, classes="table table-sm") -> str:
    return df.head(max_rows).to_html(classes=classes, index=False, escape=False)

def _describe_df(df: pd.DataFrame) -> pd.DataFrame:
    desc_num = df.describe(include=[np.number]).transpose()
    desc_obj = df.describe(include=['object', 'category', 'bool']).transpose()
    combined = pd.concat([desc_num, desc_obj], axis=0, sort=False).fillna('')
    return combined

def _missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    uniq = df.nunique(dropna=False)
    types = df.dtypes
    mv = pd.DataFrame({
        "dtype": types.astype(str),
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2),
        "unique_values": uniq
    }).sort_values("missing_pct", ascending=False)
    return mv

def _top_value_counts(df: pd.DataFrame, col: str, topn:int=20) -> pd.Series:
    return df[col].fillna("Missing").value_counts().nlargest(topn)

def _plot_histograms(df: pd.DataFrame, numeric_cols: List[str], max_cols:int=12) -> str:
    if not numeric_cols:
        return None
    cols = numeric_cols[:max_cols]
    n = len(cols)
    cols_per_row = 3
    rows = math.ceil(n / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*4, rows*3))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        ax = axes[i]
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(c)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Distributions")
    return _fig_to_base64(fig)

def _plot_boxplots(df: pd.DataFrame, numeric_cols: List[str], max_cols:int=9) -> str:
    if not numeric_cols:
        return None
    cols = numeric_cols[:max_cols]
    n = len(cols)
    cols_per_row = 3
    rows = math.ceil(n / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*4, rows*3))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        ax = axes[i]
        sns.boxplot(x=df[c].dropna(), ax=ax)
        ax.set_title(c)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Boxplots (outlier view)")
    return _fig_to_base64(fig)

def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
    if not numeric_cols or len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)*0.4), max(4, len(numeric_cols)*0.3)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"shrink": .7})
    ax.set_title("Correlation Heatmap")
    return _fig_to_base64(fig)

def _plot_pairwise_sample(df: pd.DataFrame, numeric_cols: List[str], sample_size:int=500) -> Optional[str]:
    if len(numeric_cols) < 2:
        return None
    cols = numeric_cols[:6]
    sample = df[cols].dropna().sample(n=min(sample_size, len(df)), random_state=0)
    if sample.shape[0] < 2:
        return None
    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha":0.6})
    fig = g.fig
    return _fig_to_base64(fig)

def _top_correlations(df: pd.DataFrame, numeric_cols: List[str], topn:int=10) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    corr = df[numeric_cols].corr().abs()
    s = corr.unstack().reset_index()
    s.columns = ["var1", "var2", "corr_abs"]
    s = s[s["var1"] != s["var2"]]
    s["pair"] = s.apply(lambda r: tuple(sorted([r["var1"], r["var2"]])), axis=1)
    s = s.drop_duplicates("pair").sort_values("corr_abs", ascending=False)
    return s[["var1","var2","corr_abs"]].head(topn).reset_index(drop=True)

def _skew_kurtosis(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    s = pd.DataFrame(index=numeric_cols)
    s["skewness"] = df[numeric_cols].skew().round(3)
    s["kurtosis"] = df[numeric_cols].kurtosis().round(3)
    return s.reset_index().rename(columns={"index":"feature"})

def _outlier_iqr(df: pd.DataFrame, col: str) -> Tuple[int, float, float]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5*iqr
    ub = q3 + 1.5*iqr
    outliers = df[(df[col] < lb) | (df[col] > ub)]
    return len(outliers), lb, ub

def _plot_categorical_bar(df: pd.DataFrame, col: str, topn:int=20) -> Optional[str]:
    if col not in df.columns:
        return None
    counts = _top_value_counts(df, col, topn)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(counts))))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"Top values for {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    return _fig_to_base64(fig)

def _target_analysis(df: pd.DataFrame, target_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
    out = {}
    if target_col not in df.columns:
        return out
    if pd.api.types.is_numeric_dtype(df[target_col]):
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df[target_col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Target distribution: {target_col}")
        out["target_dist_png"] = _fig_to_base64(fig)
        if numeric_cols:
            corrs = df[numeric_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            out["target_correlations"] = corrs.reset_index().rename(columns={"index":"feature", target_col:"abs_corr"})
    else:
        counts = _top_value_counts(df, target_col, topn=50)
        out["target_counts_html"] = counts.reset_index().rename(columns={"index":target_col, target_col:"count"}).to_html(classes='table table-sm', index=False)
        if numeric_cols:
            grp = df.groupby(target_col)[numeric_cols].agg(['mean','median','std']).transpose()
            out["target_numeric_by_cat_html"] = grp.to_html(classes='table table-sm')
    return out

# -------------- INSIGHTS SECTION --------------
def _generate_insights(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    insights = []
    if df.isnull().values.any():
        missing = df.isnull().sum()
        top_missing = missing[missing > 0].sort_values(ascending=False).head(3)
        insights.append(f"Missing values detected — top columns with missing data: {', '.join(top_missing.index)}.")
    else:
        insights.append("No missing values detected across the dataset.")

    if numeric_cols:
        corr = df[numeric_cols].corr().abs()
        top_corr = corr.unstack().sort_values(ascending=False).drop_duplicates().head(3)
        top_corr = [(i[0], i[1], round(v, 2)) for i, v in top_corr.items() if i[0] != i[1]]
        if top_corr:
            corr_text = ", ".join([f"{a}-{b}: {c}" for a, b, c in top_corr])
            insights.append(f"Highest correlated numeric features: {corr_text}.")
        skew = df[numeric_cols].skew().abs()
        high_skew = skew[skew > 1].index.tolist()
        if high_skew:
            insights.append(f"Highly skewed features (possible transformation candidates): {', '.join(high_skew[:5])}.")
    if categorical_cols:
        cat = categorical_cols[0]
        top_val = df[cat].value_counts().idxmax()
        insights.append(f"The most frequent category in '{cat}' is '{top_val}'.")
    insights.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.")
    return insights

# -------------- MAIN FUNCTION --------------
def generate_eda_context(csv_path: Optional[str]=None,
                         df: Optional[pd.DataFrame]=None,
                         target_col: Optional[str]=None,
                         selected_categorical: Optional[str]=None,
                         max_hist_cols:int=12) -> Dict[str, Any]:

    if df is None and csv_path is None:
        raise ValueError("Provide either csv_path or df.")
    if df is None:
        df = pd.read_csv(csv_path)

    row_count = len(df)
    col_count = df.shape[1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    sample_html = _df_to_html(df, max_rows=20)
    describe_html = _describe_df(df).to_html(classes='table table-sm')
    missing_html = _missing_values_table(df).to_html(classes='table table-sm')

    top_values_htmls = {}
    for col in (categorical_cols[:6]):
        vals = _top_value_counts(df, col, topn=20).reset_index().rename(columns={"index":col, col:"count"})
        top_values_htmls[col] = vals.to_html(classes='table table-sm', index=False)

    hist_png = _plot_histograms(df, numeric_cols, max_cols=max_hist_cols)
    box_png = _plot_boxplots(df, numeric_cols)
    corr_png = _plot_correlation_heatmap(df, numeric_cols)
    pair_png = _plot_pairwise_sample(df, numeric_cols)

    skew_kurt_html = _skew_kurtosis(df, numeric_cols).to_html(classes='table table-sm') if numeric_cols else ""
    outlier_summary = []
    for c in numeric_cols[:20]:
        num_out, lb, ub = _outlier_iqr(df, c)
        outlier_summary.append({"feature": c, "num_outliers": num_out, "lower_bound": round(lb,3), "upper_bound": round(ub,3)})
    outlier_html = pd.DataFrame(outlier_summary).to_html(classes='table table-sm', index=False)

    cat_col = selected_categorical if (selected_categorical in df.columns) else (categorical_cols[0] if categorical_cols else None)
    cat_png = _plot_categorical_bar(df, cat_col) if cat_col else None

    top_corrs_html = _top_correlations(df, numeric_cols, topn=15).to_html(classes='table table-sm', index=False)
    target_result = _target_analysis(df, target_col, numeric_cols) if target_col else {}

    # ✅ FIXED INSIGHTS SECTION
    insights_list = _generate_insights(df, numeric_cols, categorical_cols)
    insights_html = "<ul>" + "".join([f"<li>{i}</li>" for i in insights_list]) + "</ul>"

    context = {
        "row_count": row_count,
        "col_count": col_count,
        "sample_html": sample_html,
        "describe_html": describe_html,
        "missing_html": missing_html,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "top_values_htmls": top_values_htmls,
        "hist_png": hist_png,
        "box_png": box_png,
        "corr_png": corr_png,
        "pair_png": pair_png,
        "skew_kurt_html": skew_kurt_html,
        "outlier_html": outlier_html,
        "cat_png": cat_png,
        "cat_col": cat_col,
        "top_corrs_html": top_corrs_html,
        "target_analysis": target_result,
        "insights_html": insights_html,
        "insights": insights_list,  # ✅ ADDED this line (fixes the 'undefined' error)
    }
    return context

def save_base64_png(data_uri: str, filename: str):
    assert data_uri.startswith("data:image/png;base64,")
    b = base64.b64decode(data_uri.split(",",1)[1])
    with open(filename, "wb") as f:
        f.write(b)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA and produce context dictionary (prints keys).")
    parser.add_argument("--csv", required=True, help="CSV path")
    parser.add_argument("--target", help="Target column (optional)")
    args = parser.parse_args()
    ctx = generate_eda_context(csv_path=args.csv, target_col=args.target)
    print("Generated keys:", list(ctx.keys()))
