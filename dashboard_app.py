import streamlit as st
import psutil

st.title("Federated Learning - Context Monitoring Dashboard")

cpu = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory().percent

st.metric(label="CPU Usage (%)", value=f"{cpu}%")
st.metric(label="Memory Usage (%)", value=f"{memory}%")

if cpu > 70 and memory > 70:
    st.success("Choosing SMALL model")
else:
    st.success("Choosing LARGE model")