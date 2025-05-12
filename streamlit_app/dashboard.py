import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from flask import session, flash, redirect, url_for
import json
from urllib.parse import urlparse, parse_qs
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np

st.set_page_config(page_title="📊 Data Dashboard", layout="wide")

query_params = st.experimental_get_query_params()
user_id = query_params.get("user_id", [None])[0]

cleaned_file_path = None

if user_id:
    context_path = f"streamlit_app/context_{user_id}.json"
    try:
        with open(context_path, "r") as f:
            context = json.load(f)
            cleaned_file_path = context.get("cleaned_data_path")
    except (FileNotFoundError, json.JSONDecodeError):
        cleaned_file_path = None

# cleaned_file_path = session.get('cleaned_data_path')

st.title("📈 Business Data Visualization")

if cleaned_file_path and os.path.exists(cleaned_file_path):
    df = pd.read_csv(cleaned_file_path)
    st.success("✅ Cleaned data loaded!")

    date_cols = df.select_dtypes(include=['object', "datetime64"]).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            continue

    selected_col = st.selectbox("🔎 Select a column to visualize", df.columns)

    # === Advanced Filters ===
    st.sidebar.header("🔧 Data Filters")

    # Multi-select filter for categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            unique_vals = df[cat_col].dropna().unique().tolist()
            selected_vals = st.sidebar.multiselect(
                f"Filter by {cat_col}", 
                unique_vals, default=unique_vals
            )
            df = df[df[cat_col].isin(selected_vals)]
    
    # Date range filter
    datetime_cols = df.select_dtypes(include='datetime64[ns]').columns
    if len(datetime_cols) > 0:
        date_col = st.sidebar.selectbox(
            "📅 Filter by date column", 
            datetime_cols
        )
        min_date, max_date = df[date_col].min(), df[date_col].max()
        start_date, end_date = st.sidebar.date_input(
            "Select date range", 
            [min_date, max_date]
        )
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]
    
    chart_type = st.selectbox("Select chart type", 
        ["Histogram", "Pie Chart", "Scatter Plot", "Line Chart", "Bubble Chart", "Multi-Histogram", "Correlation Matrix"]
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if selected_col:
        # col_data = filtered_df[selected_col]
        col_data = df[selected_col]
        dtype = col_data.dtype

        if chart_type == "Histogram" and pd.api.types.is_numeric_dtype(dtype):
            st.subheader("📉 Interactive Histogram")
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
            fig = px.histogram(
                df, x=selected_col, nbins=bins, title=f"Distribution of {selected_col}",
                color_discrete_sequence=['indianred']
            )
            fig.update_layout(
                xaxis_title=selected_col,
                yaxis_title="Frequency",
                bargap=0.1,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Multi-Histogram":
            selected_cols = st.multiselect("Select multiple numeric columns", numeric_cols)
            for col in selected_cols:
                fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie Chart" and col_data.nunique() <= 20:
            st.subheader("🥧 Interactive Pie Chart")
            value_counts = col_data.value_counts().reset_index()
            value_counts.columns = [selected_col, "count"]
            fig = px.pie(value_counts, names=selected_col, values="count", title=f"Distribution of {selected_col}",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textinfo="percent+label", pull=[0.05]*len(value_counts))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            st.subheader("🔹 Scatter Plot")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
                             color_discrete_sequence=['seagreen'],
                             hover_data=df.columns)
            fig.update_layout(hovermode="closest", xaxis_title=x_col, yaxis_title=y_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart":
            st.subheader("📈 Line Chart")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            x_col = st.selectbox("X-axis (typically time)", df.columns)
            y_col = st.selectbox("Y-axis (numeric)", numeric_cols)
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}",
                          markers=True, line_shape="spline", render_mode="svg")
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Bubble Chart":
            x = st.selectbox("X-axis", numeric_cols, key="bubble_x")
            y = st.selectbox("Y-axis", numeric_cols, key="bubble_y")
            size = st.selectbox("Bubble Size", numeric_cols, key="bubble_size")
            color = st.selectbox("Color by", df.columns, key="bubble_color")
            hover = st.selectbox("Hover Name", df.columns, key="bubble_hover")

            fig = px.scatter(
                df, x=x, y=y, size=size, color=color,
                hover_name=hover, size_max=60,
                title=f"{x} vs {y} with bubble size {size}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # === Correlation Matrix ===
        elif chart_type == "Correlation Matrix":
            st.subheader("📊 Correlation Matrix (Numeric Columns Only)")
            numeric_df = df.select_dtypes(include='number')

            if not numeric_df.empty and numeric_df.shape[1] >= 2:
                corr_matrix = numeric_df.corr()

                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    title="Correlation Matrix"
                )
                fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns to compute correlation matrix.")
        else:
            st.warning("Chart type not supported for selected column type or too many categories.")
        
    else:
        st.error("⚠️ Cleaned file not found. Please upload and preprocess a dataset in upload page.")