"""Streamlit frontend for the Financial Reasoning Agent."""

import streamlit as st

st.set_page_config(page_title="Financial Agent", layout="wide")
st.title("Autonomous Financial Reasoning Agent")
st.write("Ask a question about financial data and the agent will reason through it step by step.")

question = st.text_input("Enter your financial question:")

if question:
    st.info("Agent processing... (not yet connected)")
    # TODO: call API endpoint and display reasoning trace
