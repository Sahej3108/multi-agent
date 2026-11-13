from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
import os
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="CodeMind - Error Detection", page_icon="🛠️", layout="wide")

# -----------------------------
# LLM Configuration
# -----------------------------
def llm_result():
    return ChatGroq(
        model="openai/gpt-oss-20b",  # Updated to a standard supported Groq model
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

# Initialize LLM safely
if "llm" not in st.session_state:
    try:
        st.session_state.llm = llm_result()
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.session_state.llm = None

st.title("🛠️ CodeMind : Agent 1 - Error Detection")
st.write("Paste your code, choose the programming language, and generate an error-free version using Groq LLM.")

languages = ["Python", "Java", "C", "C++", "JavaScript", "TypeScript", "Kotlin", "Swift", "Go", "Rust", "PHP", "R", "SQL"]

language = st.selectbox("Programming Language", languages)

code = st.text_area("Paste Your Code", height=350, placeholder="Paste your source code here...")

# Initialize session state for output tracking
if "analysis_output" not in st.session_state:
    st.session_state.analysis_output = None

col1, col2 = st.columns([1, 5])

with col1:
    analyze = st.button("🔍 Analyze", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

# -----------------------------
# AGENT 1 : ERROR DETECTION FUNCTION
# -----------------------------
def error_detection_agent(code_snippet: str, lang: str = "Python"):
    prompt = f"""
You are Agent 1 of CodeMind: Error Detection Agent.

Analyze the following {lang} code and identify:

1. Syntax Errors
2. Runtime Risks
3. Logical Bugs
4. Bad Coding Practices

Instructions:
- Mention line numbers whenever possible.
- Explain why the issue occurs.
- Suggest the corrected code.
- If no errors exist, state "No major errors detected."

Source Code:
```{lang.lower()}
{code_snippet}
```
"""
    response = st.session_state.llm.invoke(prompt)
    st.write("DEBUG raw response:", response)   # temp
    st.write("DEBUG content:", repr(response.content))  # temp
       
    return response.content

   
  

# -----------------------------
# Button Logic
# -----------------------------
if clear:
    st.session_state.analysis_output = None
    code = ""
    st.rerun()

if analyze:
    if not code.strip():
        st.warning("⚠️ Please paste some code before analyzing.")
    elif st.session_state.get("llm") is None:
        st.error("LLM is not initialized. Check your GROQ_API_KEY in your .env file.")
    else:
        with st.spinner("Analyzing your code..."):
            try:
                st.session_state.analysis_output = error_detection_agent(code, language)
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# -----------------------------
# Display Output
# -----------------------------
if st.session_state.analysis_output:
    st.subheader("📋 Analysis Result")
    st.markdown(st.session_state.analysis_output)

