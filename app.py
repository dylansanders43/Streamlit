import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Benford's Law Dashboard")

# 1. Upload a CSV or Excel file
file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if file is None:
    st.stop()
df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

# 2. Filters (three criteria)
st.sidebar.header("Filters")
num_cols = df.select_dtypes("number").columns.tolist()
if not num_cols:
    st.error("The file needs at least one numeric column.")
    st.stop()

# Filter 1: category values
cat_col = st.sidebar.selectbox("Filter 1: category column", df.columns)
cat_vals = st.sidebar.multiselect("Keep these values (empty = all)", df[cat_col].dropna().unique())
if cat_vals:
    df = df[df[cat_col].isin(cat_vals)]

# Filter 2: numeric range
rng_col = st.sidebar.selectbox("Filter 2: numeric column", num_cols)
lo, hi = float(df[rng_col].min()), float(df[rng_col].max())
if lo < hi:
    lo, hi = st.sidebar.slider("Range", lo, hi, (lo, hi))
    df = df[df[rng_col].between(lo, hi)]

# Filter 3: text search
txt_col = st.sidebar.selectbox("Filter 3: text search column", df.columns)
txt = st.sidebar.text_input("Contains (empty = all)")
if txt:
    df = df[df[txt_col].astype(str).str.contains(txt, case=False)]

st.write(f"Rows after filtering: {len(df)}")
st.dataframe(df)

# 3. Benford analysis
st.header("Benford Analysis")
col = st.selectbox("Column to analyze", num_cols)
k = 1 if st.radio("Digits", ["First digit", "First two digits"]) == "First digit" else 2

vals = df[col].dropna().abs()
vals = vals[vals > 0]
n = len(vals)
if n == 0:
    st.warning("No positive values to analyze.")
    st.stop()

digits = np.floor(vals / 10 ** (np.floor(np.log10(vals)) - (k - 1))).astype(int)
labels = np.arange(10 ** (k - 1), 10 ** k)
actual = digits.value_counts().reindex(labels, fill_value=0).values / n
benford = np.log10(1 + 1 / labels)

table = pd.DataFrame({"Digit": labels, "Actual": actual, "Benford": benford})

# 4. KS statistic (numpy)
D = np.max(np.abs(np.cumsum(actual) - np.cumsum(benford)))
critical = 1.36 / np.sqrt(n)  # critical value at alpha = 0.05
p_value = min(1.0, 2 * np.exp(-2 * n * D ** 2))  # approximate p-value

# 5. Visualization
fig = px.bar(table, x="Digit", y=["Actual", "Benford"], barmode="group",
             title=f"Actual vs Benford distribution: {col}")
st.plotly_chart(fig)

# 6. Statistical results
st.subheader("KS Test Results")
st.write(f"KS statistic (D): {D:.4f}")
st.write(f"Critical value (alpha = 0.05): {critical:.4f}")
st.write(f"Approximate p-value: {p_value:.4f}")
if D > critical:
    st.error("Statistically different from the Benford distribution.")
else:
    st.success("Not statistically different from the Benford distribution.")

# 7. Downloads
st.header("Downloads")
st.download_button("Summary statistics (CSV)", df.describe().to_csv(), "summary_stats.csv")
st.download_button("Benford results (CSV)", table.to_csv(index=False), "benford_results.csv")
st.download_button("Chart (HTML)", fig.to_html(), "benford_chart.html")
