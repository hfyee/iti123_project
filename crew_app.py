import streamlit as st
import os
from crewai import Crew, Process

st.set_page_config(page_title="AI Research Assistant", page_icon="🦋", layout="wide")
st.title("🦋 AI Research Assistant")
st.markdown("Enter your product category and let the bot conduct market research, analysis, and generate a report for you.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")

    st.divider()
    st.info("Version v0.1.0")


# Check for keys
if not openai_api_key or not tavily_api_key:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set Environment Variables for LangChain
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

import crew_logic
from crew_logic import crew
import asyncio

#async def run_crew_async(crew: Crew, question: str):
#    return crew.kickoff(inputs={'question': question})
async def run_crew_async(crew: Crew, inputs: dict):
    return crew.kickoff(inputs=inputs)

product_category = st.text_input("Product category:", placeholder="e.g., ebikes")

if st.button("Run Task"):
    if not product_category.strip():
        st.warning("Please enter a product category.")
    else:
        with st.spinner("Router analyzing and assigning task..."):
            # Run the crew
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_crew_async(crew, inputs={'product_category': product_category}))

        st.success("✅ Task Completed!")
        st.markdown("### ✨ Result:")
        st.write(result.raw)