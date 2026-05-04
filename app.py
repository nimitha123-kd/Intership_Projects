import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# --- INITIALIZATION ---
st.set_page_config(page_title="AI SQL Analyst Agent", layout="wide")
st.title("🚀 AI SQL Data Analyst (CSV ➔ SQL ➔ Insights)")

# --- 1. FILE UPLOAD & DB CONVERSION ---
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if uploaded_file and api_key:
    # Load CSV into Pandas
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Create temporary SQLite Database
    engine = create_engine("sqlite:///temp_data.db")
    df.to_sql("data_table", engine, index=False, if_exists="replace")
    db = SQLDatabase(engine)

    # --- 2. LLM & AGENT SETUP ---
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )
    
    # LangChain SQL Agent handles the query generation and execution
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    # --- 3. NATURAL LANGUAGE QUERY ---
    user_question = st.text_input("Ask a question about your data:")

    if user_question:
        with st.spinner("Analyzing..."):
            try:
                # Get response from the agent
                response = agent_executor.invoke({"input": user_question})
                
                # --- 4. DISPLAY RESULTS ---
                st.subheader("Final Answer")
                st.write(response["output"])

                # Optional: Show the SQL used (if you want to see the "thought" process)
                # Note: LangChain agents usually show this in logs, but we can prompt for it.
                
                # --- 5. VISUALIZATION (Simple Implementation) ---
                st.subheader("Data Visualization")
                # Logic: If the user asks for a chart, we can let Pandas handle the summary
                if "chart" in user_question.lower() or "plot" in user_question.lower():
                    st.bar_chart(df.select_dtypes(include=['number']).iloc[:, :2])
                    st.info("Tip: For complex charts, the agent describes the trend; this UI shows a general numeric overview.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file and enter your Groq API key to begin.")
