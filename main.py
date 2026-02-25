'''
Multiagent RAG system in CrewAI workflow with React UI
'''
import os
import json
import asyncio
import glob
import base64
import io
# Before importing requests, set USER_AGENT env var to identify your requests to websites
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
import requests
import yt_dlp
from pathlib import Path
from typing import Type, List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
from datetime import datetime
import queue
import threading

# Web Server & API Imports
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
from dotenv import load_dotenv
import uvicorn

# Langchain Imports
from langchain_openai import ChatOpenAI
# For RAG pipeline
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
# For tools
from langchain_tavily import TavilySearch
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.tools import tool

# CrewAI Imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow import Flow, listen, or_, start, router
from crewai.tools import BaseTool
from crewai_tools import RagTool
from crewai_tools.tools.rag import RagToolConfig, VectorDbConfig, ProviderSpec
from crewai_tools import FileWriterTool, YoutubeVideoSearchTool, CodeInterpreterTool
from crewai.tasks.task_output import TaskOutput

# Composio Imports
from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider

# Image Processing Imports
from PIL import Image
from ultralytics import YOLO
from wordcloud import WordCloud

# --- 1. Configuration & Setup ---

# Load environment variables (Create a .env file with your keys)
import gdown
url = 'https://drive.google.com/file/d/17C0MsdQ0gN9bHML_dYOQQ1CUxzIdkF0q/view?usp=drive_link' # HF
output_path = '.env'
gdown.download(url, output_path, quiet=False,fuzzy=True)
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Market Research Agent API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://research-frontend-209844603068.asia-southeast1.run.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files statically so Next.js can display generated images
OUTPUT_DIR = "output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Clear output dir if not empty
filesList = glob.glob(OUTPUT_DIR + "/*")
for file in filesList:
    os.remove(file)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# API Keys Check
REQUIRED_KEYS = ["OPENAI_API_KEY", "TAVILY_API_KEY", "COMPOSIO_API_KEY", "COMPOSIO_USER_ID", "GEMINI_API_KEY"]
for key in REQUIRED_KEYS:
    if not os.getenv(key):
        print(f"WARNING: {key} is missing from environment variables.")

# --- 2. Tool Definitions ---
# (Maintained from original, removed Streamlit dependencies)

class GenerationTool(BaseTool):
    name: str = 'Generation'
    description: str = 'Useful for general queries answered by the LLM.'
    def _run(self, query: str) -> str:
        return llm.invoke(query).content

class TavilySearchInput(BaseModel):
    query: str = Field(description="The search query.")
    search_depth: str = Field(default="basic")
    include_domains: str = Field(default="")
    include_images: bool = Field(default=False)
    include_image_descriptions: bool = Field(default=False)

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "Searches the internet using Tavily."
    args_schema: Type[BaseModel] = TavilySearchInput
    search: TavilySearch = Field(default_factory=TavilySearch)
    def _run(self, query: str, **kwargs) -> str:
        return self.search.run(query)

class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = "Search Wikipedia."
    def _run(self, query: str) -> str:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
        return api_wrapper.run(query)

class YouTubeSearchTool(BaseTool):
    name: str = "youtube"
    description: str = "Useful for when you need to search for videos on YouTube."
    search: YouTubeSearchTool = Field(default_factory=YouTubeSearchTool)

    def _run(self, query: str) -> str:
        return self.search.run(query)
    
class DallEImageTool(BaseTool):
    name: str = "dalle"
    description: str = "Generate images from text."
    def _run(self, query: str) -> str:
        api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1024x1024")
        return api_wrapper.run(query)

class YoloToolInput(BaseModel):
    image_path: str = Field(..., description="Path to the image.")

class YoloDetectorTool(BaseTool):
    name: str = "YOLO Object Detector"
    description: str = "Detects objects in images."
    args_schema: Type[BaseModel] = YoloToolInput
    def _run(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            model = YOLO('yolo11n.pt')
            results = model.predict(image, conf=0.5)
            ultralytics_results = results[0]
            labels = []
            for box in ultralytics_results.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                labels.append(class_name)
            return str(labels[0]) if labels else "unknown"
        except Exception as e:
            return f"Error detecting objects: {str(e)}"

class LowResBase64EncodingToolInput(BaseModel):
    image_path: str = Field(..., description="Path to image.")
    max_width: int = 800
    max_height: int = 800
    quality: int = 60

class LowResBase64EncodingTool(BaseTool):
    name: str = 'Image base64 encoding'
    description: str = 'Encodes image to base64.'
    args_schema: Type[BaseModel] = LowResBase64EncodingToolInput
    def _run(self, image_path: str, max_width: int=800, max_height: int=800, quality: int=85) -> str:
        img = Image.open(image_path)
        img.thumbnail((max_width, max_height))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class WordCloudToolInput(BaseModel):
    text: str = Field(..., description="Text for word cloud.")
    colormap: str = Field(description="Color scheme.")
    output_image_path: str = Field(description="Save path.")

class WordCloudGenerationTool(BaseTool):
    name: str = "Word Cloud Generator"
    description: str = "Generates word cloud image."
    args_schema: Type[BaseModel] = WordCloudToolInput
    def _run(self, text: str, colormap: str, output_image_path: str) -> str:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
        wordcloud.to_file(output_image_path)
        return f"Word cloud saved to {output_image_path}"

class CheckYouTubeLinkToolInput(BaseModel):
    youtube_url: str = Field(..., description="The YouTube video URL to check.")

class CheckYouTubeLinkTool(BaseTool):
    name: str = "YouTube Link Checker"
    description: str = "Checks if a YouTube video link is valid and accessible."
    args_schema: Type[BaseModel] = CheckYouTubeLinkToolInput

    def _run(self, youtube_url: str) -> str:
        if not ("youtube.com/watch" in youtube_url or "youtu.be/" in youtube_url):
          return "Invalid format: Not a recognized YouTube link."
    
        ydl_opts = {'quiet': True, 'no_warnings': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Try to extract video information
                info = ydl.extract_info(youtube_url, download=False)
                if info:
                    return f"Valid: {info.get('title', 'Unknown Title')}"
                else:
                    return "Unavailable: Video does not exist or is private."
        except Exception as e:
            return f"Error/Unavailable: {str(e)}"

# --- 3. Agent & Task Setup ---

# LLM Initialization
llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

vlm = LLM(
    #model="gpt-4.1",
    #api_key=os.getenv("OPENAI_API_KEY"),
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

qa_llm = LLM(
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)

# --- Custom RAG tool ---

# Document loaders
file_loaders = {
    '.pdf': PyMuPDFLoader,
    '.txt': TextLoader,
}

# Define a function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=file_loaders[file_type],
        show_progress=True
    )

# Create DirectoryLoader instances for each file type
pdf_loader = create_directory_loader('.pdf', './rag_docs')
txt_loader = create_directory_loader('.txt', './rag_docs')

# Load the documents
pdf_docs = pdf_loader.load()
txt_docs = txt_loader.load()

print(f"Loaded {len(pdf_docs)} PDF files.")
'''
for doc in pdf_docs:
    print(f"Metadata:\n{doc.metadata}\n")
    print(f"Content snippet:\n{doc.page_content[:100]}...\n")
'''
print(f"Loaded {len(txt_docs)} text files.")
'''
for doc in txt_docs:
    print(f"Metadata:\n{doc.metadata}\n")
    print(f"Content snippet:\n{doc.page_content[:100]}...\n")
'''
# Website loader
urls = [
    "https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html",
    "https://en.wikipedia.org/wiki/Folding_bicycle"
]

web_loader = WebBaseLoader(urls)

# Load the documents
web_docs = web_loader.load()

print(f"Loaded {len(web_docs)} URLs.")
'''
for doc in web_docs:
    print(f"Metadata:\n{doc.metadata}\n")
    print(f"Content snippet:\n{doc.page_content[:100]}...\n")
'''
# Split documents into chunks
# chunk_size=500, chunk_overlap ~10% of chunk_size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

all_docs = web_docs + txt_docs + pdf_docs

split_docs = text_splitter.split_documents(all_docs)

# Pinecone vector store
index_name = "iti123-openai-index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'), dimensions=512)

# Initialize vector store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Add documents
vectorstore.add_documents(split_docs)

# Retrieve relevant documents
retriever = vectorstore.as_retriever(search_type="similarity")
'''
# Set up RetrievalQA chain using the corrected qa_llm
qa = RetrievalQA.from_chain_type(llm=qa_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Perform similarity search
queries = [
    "Why is frame and build quality important for an e-bike purchase?",
    "Why do e-scooter riders face issues in safely signalling turns?"
]

query = queries[1]
#print(vectorstore.similarity_search(query, k=3))
result = qa.invoke({"query": query})
print("QA Response:", result)
'''
class PineconeRerankTool(BaseTool):
    name: str = "Pinecone Advanced RAG Search"
    description: str = (
        "Useful for retrieving specific information from the internal knowledge base "
        "stored in Pinecone with reranking. Use this to find documents, context, or past data."
    )

    index_name: str = "iti123-openai-index"

    def _run(self, query: str) -> str:
        # Initialize embeddings (must match what you used to ingest data)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'), dimensions=512)

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings,
        )

        # Define base retriever (fetch more docs initially, e.g. 5-10)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Initialize reranker (return top_n relevant documents)
        compressor = PineconeRerank(model="bge-reranker-v2-m3", top_n=2)

        # Create compression pipeline
        rerank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        # Execute search
        results = rerank_retriever.invoke(query)

        # Format the output for the Agent
        formatted_results = "\n\n".join([doc.page_content for doc in results])

        return f"Retrieved Context:\n{formatted_results}"

# Tool Instances
rag_tool = PineconeRerankTool(index_name="iti123-openai-index")
generation_tool = GenerationTool()
web_search_tool = TavilySearchTool(search_depth="basic", include_images=True)
shopping_web_search_tool = TavilySearchTool(search_depth="advanced", include_images=True)
wiki_tool = WikipediaTool()
dalle_tool = DallEImageTool()
youtube_tool = YouTubeSearchTool()
youtube_rag_tool = YoutubeVideoSearchTool(summarize=True)
file_writer_tool = FileWriterTool(directory=OUTPUT_DIR)
code_interpreter = CodeInterpreterTool()
encode_image_base64 = LowResBase64EncodingTool()
obj_detector_tool = YoloDetectorTool()
word_cloud_tool = WordCloudGenerationTool()
check_youtube_link_tool = CheckYouTubeLinkTool()

# Composio Setup
composio_tools = []
if os.getenv("COMPOSIO_API_KEY"):
    composio = Composio(provider=OpenAIAgentsProvider(), api_key=os.getenv("COMPOSIO_API_KEY"))
    composio_tools = composio.tools.get(user_id=os.getenv("COMPOSIO_USER_ID"), tools=["reddit"])

# Callback (Modified for API - no Streamlit)
def word_cloud_callback(output: TaskOutput):
    """Callback to log completion of word clouds."""
    print(f"### Sentiment analysis completed. Check {OUTPUT_DIR}/ for generated images. ###")

# Function-based guardrail for grading task output
def validate_eval_content(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate evaluation content meets requirements."""
    try:
        # Check word count
        word_count = len(result.raw.split())
        if word_count > 300:
            return (False, "Evaluation content exceeds 300 words")

        # Additional validation logic here
        return (True, result.raw.strip())
    except Exception as e:
        return (False, "Unexpected error during validation")

# --- Agents Definition ---
video_researcher = Agent(
    role="Video Researcher",
    goal="Extract info from YouTube videos",
    backstory='Expert researcher specializing in video content.',
    tools=[youtube_rag_tool, file_writer_tool],
    max_iter=5,
    verbose=True,
    llm=llm
)

reddit_researcher = Agent(
    role="Reddit Search Assistant",
    goal="Search Reddit for consumer feedback",
    backstory="Assistant with access to Reddit search tools.",
    tools=composio_tools,
    verbose=True,
    llm=llm
)

analyst = Agent(
    role='Senior Research Analyst',
    goal="Conduct market research.",
    backstory="""An experienced research analyst with expertise in identifying
    market trends and opportunities as well as understanding consumer behavior.
    You always starts with a foundational understanding from Wikipedia before 
    diving deeper. """,
    tools=[rag_tool, wiki_tool, web_search_tool, youtube_tool],
    allow_delegation=True,
    max_iter=7,
    verbose=True,
    llm=llm
)

writer = Agent(
    role='Report Writer',
    goal="""Create comprehensive, well-structured reports combining the provided
    research and news analysis. Do not include any information that is not explicitly
    provided.""",
    backstory="""A professional report writer with experience in business intelligence 
    and market analysis. You have an MBA and excel at synthesizing information into 
    clear and actionable insights.""",
    allow_delegation=True,
    max_iter=7,
    verbose=True,
    llm=llm
)

# -----------------------------
# Quality assurance
# -----------------------------
editor = Agent(
    role="Content Editor",
    goal="Ensure content quality and consistency.",
    backstory="""An experienced editor with an eye for detail. You excel at
    critiquing market research and competive analysis reports, ensuring content
    meets high standards for clarity and accuracy.""",
    tools=[rag_tool, wiki_tool, youtube_rag_tool, file_writer_tool],
    allow_delegation=True,
    max_iter=7,
    verbose=True,
    llm=llm
)

content_evaluator = Agent(
    role="Content Evaluator",
    goal="Evalue market analysis report from your Marketing team.",
    backstory="""Product Marketing Manager for {product} company. You have years
    of experience at a top market research consultancy, specialized in gleaning
    data-driven insights to guide business strategy and consumer understanding.""",
    tools=[rag_tool, wiki_tool, youtube_rag_tool],
    allow_delegation=False,
    max_iter=5,
    verbose=True,
    llm=qa_llm,
)

image_analyst = Agent(
    role='Visual Data Specialist',
    goal='Analyze product image and create variant',
    backstory="""Expert in computer vision, capable of interpreting complex
    visual data. You are also an experienced industrial designer with expertise 
    in 3D modeling (CAD), and prototyping of modern bicycles. 
    You have a gift for creating detailed prompts for DALL-E 3.""",
    multimodal=True,
    tools=[encode_image_base64, dalle_tool],
    max_iter=10,
    verbose=True,
    llm=vlm
)

shopping_bot = Agent(
    role="Shopping Specialist for folding ebikes",
    goal="Extract specifications and attributes of ebike models from e-commerce sites.",
    backstory="""Helpful shopping assistant who is an expert in product search and 
    comparing specifications and prices for consumer goods.""",
    tools=[shopping_web_search_tool],
    max_iter=10,
    llm=llm,
    verbose=True,
)

specs_analyst = Agent(
    role="Senior Sales Engineer",
    goal="Analyze and compare technical specifications of competitor products to form insights.",
    backstory="""An experienced sales engineer with expertise in creating technical proposals 
    and delivering presentations to help customers understand how to choose folding bikes 
    including power-assisted ones.""",
    #tools=[code_interpreter, file_writer_tool],
    tools=[file_writer_tool],
    #allow_code_execution=True, # need Docker-in-Docker
    max_iter=10,
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

input_guard_agent = Agent(
    role="Input Guardrail",
    goal="Ensure inputs are relevant and safe.",
    backstory="Vigilant moderator.",
    verbose=True,
    llm=llm
)

# --- Tasks Definition  ---

video_research_task = Task(
    description="""Analyze the video at {video_url}. 
    Search for the following information about the product company 
    mentioned in the YouTube video. 
    1. Vision and design goals/philosophy
    2. Technical achievement/expertise
    3. Development challenges
    4. What makes its products unique/competitive
    5. How the company strives to be innovative (e.g. culture)
    6. Collaborations
    Save to output_files/video_transcript.md""",
    expected_output="Summary of video content.",
    agent=video_researcher
)

reddit_search_task = Task(
    description="Search Reddit for feedback on {product}.",
    expected_output="Consumer sentiment analysis.",
    agent=reddit_researcher
)

visualize_sentiments_task = Task(
    description="""Generate word clouds for {product} feedback. 
    Use the 'WordCloudGenerationTool' to generate word clouds for positive feedback
    and complaints. Focus on the adjectives in the feedback. Specify 'Reds' for 
    the colormap to represent negative feedback. Specify 'PuBuGn' for the colormap 
    to represent positive feedback. Save pngs to output_files.""",
    expected_output="Word cloud images.",
    agent=reddit_researcher,
    tools=[word_cloud_tool],
    context=[reddit_search_task],
    callback=word_cloud_callback
)

# Define Pydantic model for structured task output
class CompetitiveStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

class PricingTier(str, Enum):
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    PREMIUM = "premium"

class MarketPosition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    market_share_percent: Optional[float] = Field(None, ge=0, le=100)
    rank: Optional[int] = Field(None, ge=1)
    growth_rate_percent: Optional[float] = None

class ProductFeature(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    feature_name: str
    has_feature: bool
    quality_rating: Optional[int] = Field(None, ge=1, le=5)
    notes: Optional[str] = None

class Competitor(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str
    website: Optional[str] = None
    founded_year: Optional[int] = None
    
    # Market Position
    market_position: MarketPosition
    
    # Strengths and Weaknesses
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)
    
    # Product Analysis
    product_features: List[ProductFeature] = Field(default_factory=list)
    pricing_tier: PricingTier
    avg_price: Optional[float] = Field(None, ge=0)
    
    # Strategic Assessment
    overall_threat_level: CompetitiveStrength
    innovation_strength: CompetitiveStrength
    brand_strength: CompetitiveStrength
    customer_satisfaction: Optional[float] = Field(None, ge=0, le=5)
    
    # Additional Info
    target_audience: List[str] = Field(default_factory=list)
    key_differentiators: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class CompetitiveAnalysis(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    analysis_date: datetime = Field(default_factory=datetime.now)
    industry: str
    our_company: str
    
    competitors: List[Competitor]
    
    # Market Overview
    market_size_usd: Optional[float] = Field(None, ge=0)
    market_growth_rate: Optional[float] = None
    
    # Summary Insights
    key_insights: List[str] = Field(default_factory=list)
    our_competitive_advantages: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    strategic_recommendations: List[str] = Field(default_factory=list)

market_research_task = Task(
    description="""Research the market for {product}. Include:
    1. Key market trends
    2. Product demand
    3. Market size
    4. Consumer preferences and willingness to pay
    5. Major competitors""",
    #output_pydantic=CompetitiveAnalysis,
    expected_output="Comprehensive market research JSON.",
    agent=analyst,
    output_file=f"{OUTPUT_DIR}/research.json"
)

writing_task = Task(
    description="""Write market analysis report on {product}.
    
    Target audience: Product Marketing Manager for {product} company

    Include:
        1. Key market trends
        2. Market size
        3. Any regulatory compliance requirements in Singapore
        4. Major competitors, focusing on the current top 2 competitors
        5. Competitor analysis including SWOT analysis
        6. Comparison table of key features and pricing
        7. Consumer sentiment analysis
        8. Recommendations for {product} company's R&D strategy.

    Collaborate with your teammates to ensure the report is well-researched,
    comprehensive and accurate.

    The report should be approximately 1000-1500 words in length.
    """,
    expected_output="Markdown report.",
    context=[reddit_search_task, market_research_task],
    agent=writer
)

editing_task = Task(
    description="""Review and edit the report.

    Target audience: Product Marketing Manager for {product} company

    Your review should:
    1. Check for consistency in tone and style
    2. Improve clarity and readability
    3. Ensure content is comprehensive and accurate
    4. Check that there is a clear takeaway
    5. Check for unintended biases
    6. Check if embedded video links are accessible and not private/deleted

    Use the 'FileWriterTool' to write the final content into a markdown file 
    inside the 'output_files' directory.""",
    expected_output="Final markdown report.",
    context=[writing_task],
    agent=editor,
    output_file=f"{OUTPUT_DIR}/final_report.md"
)

"""
- When do people use them? Are there particular circumstances or occasions of use?
- How does your competitor’s share of sentiment rankings stack up to yours?
"""
grading_task = Task(
    description="""Review the market analysis report {market_report_text} objectively.
    Generate a concise evaluation and overall score based on the following criteria.

    Evaluation Criteria

    Assess if the report answers these questions about your competitors:

    1. What do people like/dislike about the product/service/brand?
    2. How much would a customer spend, and why?
    3. Are they front of mind for the customer when it comes to buying or using a
       product/service?

    4. What is their unique selling proposition?
    5. What key features are offered?
    6. What technology are they using?
    7. How much does their product cost? Is that more or less than your closest equivalent?
    8. Do your competitors offer a much wider range of models than you?

    9. What new market trends are on the horizon?
    10. What is our place in the market landscape?
    11. What could we improve?

    Also check if the report has any embedded video links that are not accessible.

    Scoring Guidance

    Use the following general guidance for assigning scores:

    Score Range   Meaning
    9–10	      Excellent — comprehensive with deep analysis, insights and takeaways.
    7–8	          Addressed >80% of the questions, with generally good analysis. Professional tone and style, free from unintended bias.
    5–6	          Addressed >60% of the questions. Not a great deal of analysis. Some inconsistency in tone/style.
    1–4	          Too brief, addressed only <30% of the questions about competitors.

    """,
    context=[writing_task],
    expected_output="""A brief critique of the quality of the market analysis
    report and a score <x/10>, under 300 words.""",
    agent=content_evaluator,
    guardrail=validate_eval_content,
    guardrail_max_retries=3
)

check_topic_task = Task(
    description="""Check if {product} is about {topic}.
    Return 'ALLOWED' or 'OFF_TOPIC (check product-topic relevance)'.""",
    expected_output="ALLOWED or OFF_TOPIC (check product-topic relevance)",
    agent=input_guard_agent
)

check_new_color_task = Task(
    description="""Check if {new_color} is a valid color from this list: 
    [white, black, red, blue, green, yellow, purple, orange, brown, pink, grey].
    Return 'ALLOWED' or 'OFF_TOPIC (check color validity)'.""",
    expected_output="ALLOWED or OFF_TOPIC (check color validity)",
    agent=input_guard_agent
)

check_input_image_task = Task(
    description="""Use YOLO to detect if {image_url} contains a bicycle. 
    Return 'ALLOWED' or 'OFF_TOPIC (check original image not a bicycle)'.""",
    tools=[obj_detector_tool],
    expected_output="ALLOWED or OFF_TOPIC (check original image not a bicycle)",
    agent=input_guard_agent
)

check_video_link_task = Task(
    description="""Use CheckYouTubeLinkTool to check if the video link at 
    {video_url} is valid and accessible. If the video is private, deleted, or 
    geoblocked, return 'OFF_TOPIC (check video link not valid)'. Else 'ALLOWED'.""",
    tools=[check_youtube_link_tool],
    expected_output="ALLOWED or OFF_TOPIC (check video link validity)",
    agent=input_guard_agent
)

aggregate_checks_task = Task(
    description="If any check is OFF_TOPIC, return OFF_TOPIC. Else ALLOWED.",
    context=[check_topic_task, check_new_color_task, check_video_link_task, check_input_image_task],
    expected_output="Final status string with the reason if OFF_TOPIC.",
    agent=input_guard_agent
)

# --- 4. Flow & Logic ---

class MarketResearchFlow(Flow):
    
    @start()
    def check_guardrails(self):
        print(f"--- Flow Started ---")
        # 1. Run guard_crew
        # Dynamically build the task list. If there is no image, we don't run the image check.
        guard_tasks = [check_topic_task, check_new_color_task, check_video_link_task]
        if self.state.get("image_url") and len(self.state.get("image_url").strip()) > 0:
            guard_tasks.append(check_input_image_task)
        guard_tasks.append(aggregate_checks_task)

        guard_crew = Crew(
            agents=[input_guard_agent],
            tasks=guard_tasks,
            process=Process.sequential,
            verbose=True
        )
        
        crew_inputs = {
            "product": self.state.get("product"),
            "topic": self.state.get("topic"),
            "video_url": self.state.get("video_url"),
            "image_url": self.state.get("image_url"),
            "new_color": self.state.get("new_color")
        }
        result = guard_crew.kickoff(inputs=crew_inputs)

        # 2. Check if the output contains your rejection flag
        if "off_topic" in str(result).lower():
            # Fire the explicit error straight to the React frontend
            self.update_queue.put({
                "status": "error",
                #"message": "Your input is off-topic or invalid. Please check your inputs, and stick to the allowed domain."
                "message": str(result)
            })
            return "HALT_FLOW"
            
        return str(result)

    @router(check_guardrails)
    def route_query(self, guardrail_result):
        if guardrail_result == "HALT_FLOW":
            return "unsupported"
        
        query_type = self.state.get("query_type")
        if query_type == "Market research":
            return "research_crew"
        elif query_type == "Variant generation":
            return "image_crew"
        elif query_type == "Video transcription":
            return "video_crew"
        elif query_type == "Specs data collection":
            return "specs_crew"
        else:
            return "unsupported"

    @listen("research_crew")
    async def analyze_market(self):
        print("--- Starting Market Research Crew ---")

        research_crew = Crew(
            #agents=[reddit_researcher, analyst, writer, editor],
            agents=[reddit_researcher, analyst, writer],
            #tasks=[reddit_search_task, visualize_sentiments_task, market_research_task, writing_task, editing_task],
            tasks=[reddit_search_task, visualize_sentiments_task, market_research_task, writing_task],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=True,
            output_log_file=f"{OUTPUT_DIR}/research_crew_log"
        )
        crew_inputs = {"product": self.state.get("product")}
        result = await research_crew.kickoff_async(inputs=crew_inputs)

        # Save main report
        with open(f"{OUTPUT_DIR}/market_report.md", "w") as f:
            f.write(result.raw)
        
        self.current_files.append(f"{OUTPUT_DIR}/market_report.md")

        # Dynamically add any images (Word Clouds) generated by the crew
        import glob
        image_files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
        self.current_files.extend(image_files)
    
        # FIRE TO FRONTEND: Send the partial payload immediately
        self.update_queue.put({
            "status": "partial",
            "data": { "files": list(self.current_files) }
        })

        return result.raw # Pass this text directly to the next listener
    
    @listen(analyze_market)
    async def evaluate_report(self, market_report_text):
        print("--- Starting Content Evaluation Crew ---")
        evaluation_crew = Crew(
            agents=[content_evaluator],
            tasks=[grading_task],
            memory=True,
            verbose=True,
            output_log_file=f"{OUTPUT_DIR}/evaluation_crew_log"
        )
        crew_inputs = {
            "product": self.state.get("product"), "market_report_text": market_report_text
        }
        result = await evaluation_crew.kickoff_async(inputs=crew_inputs)

        # Save result to Markdown for Frontend Rendering
        with open(f"{OUTPUT_DIR}/evaluation_summary.md", "w") as f:
            f.write(result.raw)
        
        self.current_files.append(f"{OUTPUT_DIR}/evaluation_summary.md")

        # FIRE TO FRONTEND: Send the completed payload
        self.update_queue.put({
            "status": "completed",
            "data": { "files": list(self.current_files) }
        })

        # Keep a simple return for CrewAI's internal state management
        return str(result)

    @listen("video_crew")
    async def transcribe_video(self):
        print("--- Starting Video Research Crew ---")
        video_crew = Crew(
            agents=[video_researcher],
            tasks=[video_research_task],
            process=Process.sequential,
            verbose=True,
            output_log_file=f"{OUTPUT_DIR}/video_crew_log"
        )
        crew_inputs = {"video_url": self.state.get("video_url")}
        result = await video_crew.kickoff_async(crew_inputs)
        status_value = "fail" if "unable to retrieve the transcript" in str(result) else "success" 

        self.update_queue.put({
            "status": "completed",
            "data": {
                "status": status_value,
                "result": str(result), 
                "files_generated": [f"{OUTPUT_DIR}/video_transcript.md"]
            }
        })

        # Keep a simple return for CrewAI's internal state management
        return str(result)

    @listen("image_crew")
    async def generate_variant(self):
        if os.path.exists(self.state['image_url']):
            print("--- Starting Image Variant Crew ---")
            # Dynamic task definition for image generation
            generate_image_variant_task = Task(
                description=f"""
                1. Analyze the input image at {self.state['image_url']} to understand its design language.
                2. Generate a DALL-E 3 prompt for a {self.state['new_color']} version.
                Only change the color of the bike frame to {self.state['new_color']}, maintaining all 
                other aspects exactly as they are in the original image. 
                You MUST NOT add any additional accessories.
                3. Use the 'dalle' tool to generate the image.
                4. Return the image URL.
                """,
                expected_output="A URL to the generated image.",
                result_as_answer=True,
                agent=image_analyst
            )
        
            image_crew = Crew(
                agents=[image_analyst],
                tasks=[generate_image_variant_task],
                process=Process.sequential,
                planning=True,
                memory=True,
                verbose=True,
                output_log_file="output_files/image_crew_log"
            )
            crew_inputs = {"image_url": self.state.get("image_url")}
            result = await image_crew.kickoff_async(crew_inputs)
        
            # Download logic (adapted for API)
            image_url = str(result)
            save_path = f"{OUTPUT_DIR}/generated_variant_{self.state['new_color']}.jpg"

            try:
                # Simple heuristic to find URL in text response if wrapped in text
                if "http" in image_url:
                    import re
                    url_match = re.search(r'(https?://[^\s]+)', image_url)
                    if url_match:
                        image_url = url_match.group(0).strip(')"')
            
                response = requests.get(image_url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)

                    self.update_queue.put({
                        "status": "completed",
                        "data": {
                            "status": "success",
                            "result": "Image generated successfully", 
                            "files_generated": [save_path]
                        }
                    })
                    # Internal return for CrewAI
                    return "Image generated successfully"
                
                else:
                    raise Exception(f"HTTP Status {response.status_code}")

            except Exception as e:
                self.update_queue.put({
                    "status": "error",
                    "data": {
                        "status": "fail",
                        "result": f"Failed to download image: {str(e)}",
                        "files_generated": []
                    }
                })
                return "Image generation failed"

        else:
            print(f"Warning: Local file {self.state['image_url']} not found.")

    @listen("specs_crew")
    async def collect_specs(self):
        print("--- Starting Specs Collection Crew ---")
        # Parse the input string into a list
        raw_input = self.state.get('models', '')
        # Split by comma and strip whitespace: "A, B" -> ["A", "B"]
        original_list = [m.strip() for m in raw_input.split(',') if m.strip()]
        # Remove any duplicates
        models_list = list(dict.fromkeys(original_list))

        if not models_list:
            return {"error": "No models provided"}
        
        print(f"Collecting specs for models: {models_list}")

        # Update the state with the actual list so the Crew can use it
        self.state['models_list'] = models_list

        # Dynamic task definition for specs collection
        shopping_task = Task(
            description=f"""Find technical information on the bikes in this list: 
            {self.state['models_list']}.

            Extract their technical specifications. Include:
            1. An image of the bike
            2. Wheel size
            3. Number of gears
            4. Weight (in kg)
            5. Type of brakes
            6. Folded height (in cm)
            7. Folded length (in cm)
            8. Electric drivetrain (Motor, Battery, Range, Charger) ー for ebikes

            As an example, Tern Vektron specifications can be found at:
            https://www.ternbicycles.com/en/bikes/474/vektron-p10#tech_specs

            Use the 'TavilySearchTool' and set 'include_domains' to the following URLs.
            Extract the retail selling price if it is listed on these official websites.

            'include_domains': ["https://ekolife.asia/shop/",
                "https://www.decathlon.sg/c/cycling/all-bikes/folding-bikes.html",
                "https://www.decathlon.co.uk/sports/cycling/folding-bikes",
                "https://www.ternbicycles.com/en/bikes",
                "https://sg.brompton.com/c/bikes",
                "https://www.brompton.com/",
                "https://dahon.com/production?type=E-Bikes"]

            Compile your findings and organize the data in a structured format.

            If you are not sure, leave the field blank. Do not fabricate any data.
            """,
            expected_output="""Specifications of the different folding ebikes in the
            following JSON format:
            {'model_name': model_name, 'image': image_url, 'price': price,
            'specs': {'wheel_size': wheel_size, 'weight': weight, 'gears': gears}
            }
            """,
            agent=shopping_bot,
            output_file=f"{OUTPUT_DIR}/specs_data.json"
        )

        visualize_specs_task = Task(
            description="""                                                                 │
│           1. Locate the JSON file at 'output_files/specs_data.json'.
            2. Generate visualization for the JSON data such as comparison table.  
            Break down the ebike specifications to help the user compare them based on product 
            attributes such as weight, range, and battery capacity.
            3. Discuss how the different ebike models differentiate, and provide insights for consumer 
            decision making.
            """,
            expected_output="""A well-structured and insightful comparison of the ebike specifications 
            from the JSON data.""",
            context=[shopping_task],
            agent=specs_analyst,
        )

        specs_crew = Crew(
            agents=[shopping_bot, specs_analyst],
            tasks=[shopping_task, visualize_specs_task],
            process=Process.sequential,
            planning=True,
            memory=False,
            verbose=True,
            output_log_file=f"{OUTPUT_DIR}/specs_crew_log"
        )
        result = await specs_crew.kickoff_async()

        # Save result to Markdown for Frontend Rendering
        summary_filename = "specs_summary.md"
        summary_path = os.path.join(OUTPUT_DIR, summary_filename)
        
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(str(result))
        except Exception as e:
            print(f"Error saving specs summary: {e}")

        self.update_queue.put({
            "status": "completed",
            "data": {
                "status": "success",
                "result": str(result), 
                "files_generated": [summary_path]
            }
        })

        # Keep a simple return for CrewAI's internal state management
        return str(result)

    @listen("unsupported")
    def exit_flow(self):
        print("--- Guardrail triggered: Flow safely terminated ---")
        return {"Workflow halted"}

# --- 5. API Endpoints ---

class ResearchRequest(BaseModel):
    query_type: Literal['Market research', 'Variant generation', 'Video transcription', 'Specs data collection']
    product: str = "Tern folding bike"
    models: str = ""
    video_url: str = ""
    image_url: str = ""
    new_color: str = ""
    topic: str = "Folding bicycles"

    @field_validator('query_type', 'product', 'topic')
    def check_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Saves an uploaded image and returns its absolute path."""
    try:
        # Save it to our existing OUTPUT_DIR
        file_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Return the absolute path so the Crew agent can find it locally
        return {"file_path": file_path}
    except Exception as e:
        return {"error": f"Failed to upload file: {str(e)}"}

# Streaming model (Server-Sent Events) for progressive UI rendering
@app.post("/api/run-flow")
async def run_flow(request: ResearchRequest):
    """
    Main endpoint to trigger the Agentic Flow.
    Next.js should call this endpoint with JSON body.
    """
    q = queue.Queue()

    # 1. Convert request to dict for Crew inputs
    inputs = request.model_dump()
    
    # Ensure image_url is not null to prevent "NoneType" error in templates
    if not inputs.get('image_url'):
        inputs['image_url'] = ""

    try:
        """ Moved to inside main flow
        # Run Guardrail
        guard_result = await guard_crew.kickoff_async(inputs=inputs)
        
        if "OFF_TOPIC" in str(guard_result):
            raise HTTPException(status_code=400, detail="Request is OFF_TOPIC or unsafe.")
        """
        def execute_flow():
            try:
                flow = MarketResearchFlow()
                flow.update_queue = q
                flow.current_files = []            
                flow.kickoff(inputs=inputs) 
            
            except Exception as e:
                q.put({"status": "error", "message": str(e)})
            finally:
                q.put(None) 

        threading.Thread(target=execute_flow).start()

        def stream_generator():
            while True:
                result = q.get()
                if result is None:
                    break
                yield f"data: {json.dumps(result)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        # Print the actual error to your console for easier debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)