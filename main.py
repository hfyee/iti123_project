import streamlit as st
import os
import json
import asyncio
# USER_AGENT must be set before any crewai or langchain imports
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"

# Import langchain packages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# For RAG pipeline
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
# For agent tools
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.tools import tool

# Import crewai packages
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow import Flow, listen, or_, start, router, human_feedback, HumanFeedbackResult
from crewai.tools import BaseTool
from crewai_tools import RagTool
from crewai_tools.tools.rag import RagToolConfig, VectorDbConfig, ProviderSpec
from crewai_tools import FileWriterTool
from crewai_tools import YoutubeVideoSearchTool
from crewai_tools import CodeInterpreterTool
from crewai.tasks.task_output import TaskOutput

# Import pydantic packages
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
from pathlib import Path
from typing import Type, Optional, Any, Dict, List, Literal
from enum import Enum
from datetime import datetime
import requests

# Import Composio packages
from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider

# Import YOLO model for objection detection
from PIL import Image
# Streamlit has issues with ultralytics YOLO import, so use rfdetr.
from ultralytics import YOLO
#from rfdetr import RFDETRSmall
#from rfdetr.util.coco_classes import COCO_CLASSES
# For base64 encoding
import base64 
import io
import glob

# Import package for word cloud generation
from wordcloud import WordCloud

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
composio_api_key = st.secrets["COMPOSIO_API_KEY"]
composio_user_id = st.secrets["COMPOSIO_USER_ID"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Check for API keys in environment variables
if not openai_api_key or not tavily_api_key or not composio_api_key or not composio_user_id:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["COMPOSIO_API_KEY"] = composio_api_key
os.environ["COMPOSIO_USER_ID"] = composio_user_id
os.environ['GEMINI_API_KEY'] = gemini_api_key

# --- start of agentic system code ---
# Load LLM
# Using crewai.LLM
llm = LLM(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.2, # lower temp for focused management
    #max_completion_tokens=1000
)

# For Pinecone vector stor RetrievalQA chain, use langchain_openai.ChatOpenAI
qa_llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    model_name="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.2 # lower temp for focused management
    #max_completion_tokens=1000
)

vlm = LLM(
    #model="gpt-4.1",
    #api_key=openai_api_key,
    model="gemini/gemini-2.5-flash-lite",
    api_key=gemini_api_key,
    temperature=0.2
)

# --- RAG setup ---

# RagTool with ChromaDB as vector store
# default vector db is ChromaDB
vectordb: VectorDbConfig = {
    "provider": "chromadb", # alternative: qdrant
    "config": {
        "collection_name": "bikes_docs",
        "persist_directory": "./my_vector_db"
    }
}

# default embedding model is openai/text-embedding-3-large
embedding_model: ProviderSpec = {
    "provider": "openai",
    "config": {
        "model_name": "text-embedding-3-large"
    }
}

config: RagToolConfig = {
    "vectordb": vectordb,
    "embedding_model": embedding_model,
    "top_k": 4 # default is 4
}

rag_tool = RagTool(
    name="Knowledge Base", # Documentation Tool
    description="""Use this tool to retrieve information from knowledge base about:
    - Folding bikes market outlook 
    - Regulatory requirements by LTA on electric bikes in Singapore
    - Guide on creating a competitive analysis""",
    config=config,
    summarize=True
)

# Add directory of files, use its absolute path
rag_docs_path = os.path.abspath('rag_docs')
rag_tool.add(data_type="directory", path=rag_docs_path)
# Add content from web page
rag_tool.add(data_type="website", url="https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html")
rag_tool.add(data_type="website", url="https://en.wikipedia.org/wiki/Folding_bicycle")

# Tools
class GenerationTool(BaseTool):
    name: str = 'Generation'
    description: str = 'Useful for general queries answered by the LLM.'

    def _run(self, query: str) -> str:
        return llm.invoke(query)

# TavilySearch for general web search
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool"""
    query: str = Field(description="The search query to look up on the internet.")
    search_depth: str = Field(default="basic", description="The depth of the search results to return.")
    include_domains: str = Field(default="", description="A list of domains to include in the search results.")
    include_images: bool = Field(default=False, description="Whether to include images in the search results.")
    include_image_descriptions: bool = Field(default=False, description="Whether to include image descriptions in the search results.")

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "Searches the internet using Tavily to get real-time information."
    args_schema: Type[BaseModel] = TavilySearchInput
    search: TavilySearch = Field(default_factory=TavilySearch)

    def _run(self, query: str, search_depth: str, include_domains: str, include_images: bool, include_image_descriptions: bool) -> str:
        return self.search.run(query)

class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = "A tool to search for topics on Wikipedia and return a summary of the article."

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
    description: str = "Useful for when you need to generate an image from a text prompt."

    def _run(self, query: str) -> str:
        api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1024x1024")
        return api_wrapper.run(query)

# YOLO object detection model for topic guard agent
class YoloToolInput(BaseModel):
    image_path: str = Field(..., description="URL or local path to the image.")

class YoloDetectorTool(BaseTool):
    name: str = "YOLO Object Detector"
    description: str = "Detects objects in images using YOLO."
    args_schema: Type[BaseModel] = YoloToolInput

    def _run(self, image_path: str) -> str:
        # Load image and model
        #image = Image.open(requests.get(image_path).content) # URL
        image = Image.open(image_path) # or local path
        model = YOLO('yolo11n.pt')

        # Run inference
        results = model.predict(image, conf=0.5)

        # Assuming only one image is processed and results is a list with one Results object
        # Get the first (and likely only) results object
        ultralytics_results = results[0]
        labels = []
        # Populate labels list for the LabelAnnotator
        for box in ultralytics_results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0]
            print(f"Detected: {class_name} with confidence {confidence:.2f}")
            labels.append(class_name)

        # Return label of delected object class
        return str(labels[0])
_ = ''' 
class RFDetrInput(BaseModel):
    """Input for RFDetrTool."""
    image_path: str = Field(..., description="URL or local path to the image.")

class RFDetrTool(BaseTool):
    name: str = "RF-DETR Image Analyzer"
    description: str = "Detects objects in images using RF-DETR."
    args_schema: Type[BaseModel] = RFDetrInput

    def _run(self, image_path: str) -> str:
        # Load image and model
        #image = Image.open(requests.get(image_path).content) # URL
        image = Image.open(image_path) # or local path
        model = RFDETRSmall()
        detections = model.predict(image, threshold=0.5)
        # Get the labels by mapping class IDs to class names
        labels = [
            f"{COCO_CLASSES[class_id]}"
            for class_id in detections.class_id
            ]
        # Return label of delected object class
        #return str(results.data)
        return str(labels[0])
'''
# Multimodal agent requires base64 encoding for image data
# As base64 will exceed GPT-4o TPM limit of 30K, lower image resolution before encoding.
class LowResBase64EncodingToolInput(BaseModel):
    image_path: str = Field(..., description="The path to the image file to encode.")
    max_width: int = Field(default=800, description="The maximum width to resize the image to, maintaining aspect ratio.")
    max_height: int = Field(default=800, description="The maximum height to resize the image to, maintaining aspect ratio.")
    quality: int = Field(default=60, description="The quality of the JPEG compression (0-100)."
)

class LowResBase64EncodingTool(BaseTool):
    name: str = 'Image base64 encoding'
    description: str = 'Useful for encoding an image file to a base64 string.'
    args_schema: Type[BaseModel] = LowResBase64EncodingToolInput

    def _run(self, image_path: str, max_width: int = 800, max_height: int = 800, quality: int = 60) -> str:
        # Open and resize image
        img = Image.open(image_path)
        img.thumbnail((max_width, max_height)) # PIL's thumbnail expects a tuple

        # Compress image in memory
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        img_bytes = buffer.getvalue()

        # Base64 Encode
        encoded_string = base64.b64encode(img_bytes).decode('utf-8')
        return encoded_string

# Custom tool to generate sentiments word cloud
class WordCloudToolInput(BaseModel):
    text: str = Field(description="The text to generate the word cloud from.")
    colormap: str = Field(description="The color scheme for representing the words.")
    output_image_path: str = Field(description="The path where the word cloud image will be saved.")

class WordCloudGenerationTool(BaseTool):
    name: str = "Word Cloud Generator"
    description: str = "Generates a word cloud image based on input text and saves it to a specified path."
    args_schema: Type[BaseModel] = WordCloudToolInput

    def _run(self, text: str, colormap: str, output_image_path: str) -> str:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
        wordcloud.to_file(output_image_path)
        return f"Word cloud saved to {output_image_path}"

# Instantiate tools

generation_tool = GenerationTool()
web_search_tool = TavilySearchTool(
    search_depth="basic",
    include_images = True,
    include_image_descriptions = True,
    include_domains = [
        "https://www.brompton.com/stories/design-and-engineering",
        "https://de.dahon.com/pages/technology?srsltid=AfmBOopaKrg-aASd49Nwetbyxas-XzNopsGSVhGln0IIx6IJPi1T39et",
        "https://www.straitstimes.com/paid-press-releases/dahon-v-a-revolutionary-bike-tech-pushing-a-new-frontier-in-green-mobility-20250825",
        "https://www.ternbicycles.com/en/explore/choosing-bike/tern-non-electric-bike-buyer-s-guide",
        "https://www.cyclingnews.com/bikes/reviews/"
    ])

shopping_web_search_tool = TavilySearchTool(
    search_depth="advanced",
    include_images = True,
    include_image_descriptions = True,
    include_domains = [
        "https://ekolife.asia/shop/",
        "https://www.decathlon.sg/c/cycling/all-bikes/folding-bikes.html",
        "https://www.decathlon.co.uk/sports/cycling/folding-bikes",
        "https://www.ternbicycles.com/en/bikes",
        "https://sg.brompton.com/c/bikes",
        "https://www.brompton.com/",
        "https://dahon.com/production?type=E-Bikes"
    ]
)

wiki_tool = WikipediaTool() # for quick, general topical overview, as a starting point for research
dalle_tool = DallEImageTool()
youtube_tool = YouTubeSearchTool() # web scraping on YouTube search results page
youtube_rag_tool = YoutubeVideoSearchTool(summarize=True)
file_writer_tool = FileWriterTool(directory='output_files')
code_interpreter = CodeInterpreterTool()
encode_image_base64 = LowResBase64EncodingTool()
obj_detector_tool = YoloDetectorTool()
#obj_detector_tool = RFDetrTool()
word_cloud_tool = WordCloudGenerationTool()

# Composio Reddit
# Initialize Composio toolkits
composio = Composio(provider=OpenAIAgentsProvider(), api_key=composio_api_key)
# Composio Search toolkit, more than one tool
composio_tools = composio.tools.get(user_id=composio_user_id, tools=["reddit"])

# Callback function
def show_word_clouds(output: TaskOutput):
    """Callback function to print task output immediately upon completion."""
    print(f"\n### Sentiment analysis completed ###")
    folderPath = "output_files"
    filesList = glob.glob(folderPath + "/*.png")
    for file in filesList:
        caption = 'Positive feedback word cloud' if 'positive' in file else 'Complaints word cloud'
        st.image(file, caption=caption, width=200)

# Define function to download image from DALL-E tool
def download_image(image_url, save_path="./generated_image.png"):
    """Downloads an image from a URL and saves it to a file."""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved to {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")

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
    model_config = ConfigDict(extra='forbid')
    
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

# Define agents and tasks

# -----------------------------
# Video researcher
# ----------------------------
video_researcher = Agent(
    role="Video Researcher",
    goal="Extract relevant information from YouTube videos",
    backstory='An expert researcher who specializes in analyzing video content.',
    tools=[youtube_rag_tool],
    verbose=True,
)

# Define the task more generally to process multiple videos
video_research_task = Task(
    description="""Search for the following information about the company and its
    products mentioned in the YouTube video at {video_url}.
    1. Vision and design goals/philosophy
    2. Technical achievement/expertise
    3. Development challenges
    4. What makes its products unique/competitive
    5. How the company strives to be innovative (e.g. culture)
    6. Collaborations
    """,
    expected_output="""A summary of the company's R&D strategy, collaborations,
    technical expertise, and design philosophy mentioned in the video.""",
    output_file='output_files/video_transcript.md',
    agent=video_researcher,
)

# -----------------------------
# Reddit researcher
# -----------------------------
reddit_researcher = Agent(
    role="Reddit Search Assistant",
    goal="Help users search Reddit effectively",
    backstory="A helpful assistant with access to Composio Search tools.",
    tools=composio_tools,
    llm=llm,
    verbose=True
)

reddit_search_task = Task(
    description="""Search Reddit forums to get consumer feedback on {product}.""",
    expected_output="Consumer sentiment analysis from Reddit forums",
    #output_file='./output_files/reddit.md',
    agent=reddit_researcher,
)

visualize_sentiments_task = Task(
    description="""Visualize sentiment counts provided by reddit_research task.
    Use the 'WordCloudGenerationTool' to generate word clouds for positive feedback
    and complaints. Focus on the adjectives in the feedback. Specify 'Reds' for 
    the colormap to represent negative feedback. Specify 'PuBuGn' for the colormap 
    to represent positive feedback. The word clouds should be saved in the 
    'output_files' directory.""",
    tools=[word_cloud_tool],
    context=[reddit_search_task],
    expected_output="One word cloud for postive feedback, another word cloud for complaints.",
    agent=reddit_researcher,
    callback=show_word_clouds
)

# -----------------------------
# market researcher
# -----------------------------
analyst = Agent(
    role='Senior Research Analyst',
    goal="""Conduct comprehensive market research about consumer product {product}.""",
    backstory="""An experienced research analyst with expertise in identifying
    market trends and opportunities as well as understanding consumer behavior.
    You always starts with a foundational understanding from Wikipedia before 
    diving deeper. """,
    tools=[rag_tool, wiki_tool, web_search_tool, youtube_tool],
    allow_delegation=True,
    max_iter=5,
    verbose=True,
    llm=llm,
)

# expected_output="""Output JSON with indent=4 and structured according to the 
#    CompetitiveAnalysis Pydantic model.""",
market_research_task = Task(
    description="""Research the market for {product}. Include:
    1. Key market trends
    2. Product demand
    3. Market size
    4. Consumer preferences and willingness to pay
    5. Major competitors""",
    output_pydantic=CompetitiveAnalysis,
    expected_output="""Comprehensive, well-structured market research findings that 
    are easy to synthesize into a report""",
    output_file='output_files/research.json',
    agent=analyst
)

# -----------------------------
# writer
# -----------------------------
writer = Agent(
    role='Report Writer',
    goal="""Create comprehensive, well-structured reports combining the provided
    research and news analysis. Do not include any information that is not explicitly
    provided.""",
    backstory="""A professional report writer with experience in business
    intelligence and market analysis. You have an MBA from a top school. You excel
    at synthesizing information into clear and actionable insights.""",
    allow_delegation=True,
    max_iter=5,
    verbose=True,
    llm=llm,
)

writing_task = Task(
    description="""Write a well-researched, market analysis report on consumer
    product {product}.

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
    The report content should be reviewed by editor agent.
    """,
    expected_output="""A well-structured and comprehensive report, written in
    Markdown format.""",
    context=[reddit_search_task, market_research_task],
    #output_file='output_files/draft_report.md',
    agent=writer # writer leads, but can delegate research to researcher
)

# -----------------------------
# editor (for quality assurance)
# -----------------------------
editor = Agent(
    role="Content Editor",
    goal="Ensure content quality and consistency.",
    backstory="""An experienced editor with an eye for detail. You excel at
    critiquing market research and competive analysis reports, ensuring content
    meets high standards for clarity and accuracy.""",
    tools=[rag_tool, wiki_tool, youtube_rag_tool, file_writer_tool],
    allow_delegation=True,
    max_iter=5,
    verbose=True,
    llm=llm,
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
    context=[writing_task],
    expected_output="A markdown file saved in the 'output_files' directory.",
    #output_file='output_files/final_report.md',
    agent=editor
)

# -----------------------------
# Multimodal agent for image analysis and generation
# ----------------------------
image_analyst = Agent(
    role='Visual Data Specialist',
    goal='Analyze product image and create variant',
    backstory="""An expert in computer vision, capable of interpreting complex
    visual data. You are also an experienced industrial designer with expertise 
    in sketching, 3D modeling (CAD), and prototyping of modern bicycles. 
    You have a gift for creating detailed prompts for DALL-E 3.""",
    multimodal=True,
    tools=[encode_image_base64, dalle_tool],
    allow_delegation=False,
    max_iter=10,
    llm=vlm,
    verbose=True
)

_ = '''
    2. Open the image using an image viewer or analysis tool that allows for 
    close examination of details. 
    2. Use the 'Base64EncodingTool' to encode the product image to a base64 string that you can view. 
    Analyze the image. 
'''
describe_image_task = Task(
    description="""
    1. Locate the product image file at {image_url}.
    2. Open the image using an image viewer or analysis tool that allows for close examination of details. 
    3. Examine the bike's design features such as frame shape, color, and materials used. 
    4. Identify and analyze functional aspects such as the folding mechanism, wheel size, and saddle design. 
    5. Take note of additional components like brakes, gears, and any accessories included in the image. 
    6. Document your observations, focusing on dimensions, weight specifications (if visible), 
    and overall ergonomics. 
    7. Synthesize the collected information into a detailed written description, 
    capturing both technical details and asthetic appeal to ensure a thorough 
    understanding of the product's features.
    """,
    expected_output="""An accurate and detailed description of the product in the original image.""",
    output_file='output_files/image_analysis.json',
    agent=image_analyst,
)

generate_variant_task = Task(
    description="""                                                                 │
│   Generate a DALL-E 3 prompt for a {new_color} version of the bike. 
    Only change the color of the BIKE FRAME to {new_color}, maintaining all other aspects 
    exactly as they are in the original image. You MUST NOT add any additional accessories.
    """,
    context=[describe_image_task],
    expected_output="A URL to the generated image.",
    agent=image_analyst,
    result_as_answer=True
)

generate_image_variant_task = Task(
    description="""                                                                 │
    1. Analyze the input image at {image_url} to understand its design language.
    2. Generate a DALL-E 3 prompt for a {new_color} version.
    Only change the color of the bike frame to {new_color}, maintaining all other aspects 
    exactly as they are in the original image. You MUST NOT add any additional accessories.
    """,
    expected_output="A URL to the generated image.",
    agent=image_analyst,
    result_as_answer=True
)

# -----------------------------
# Specs collection and visualization agents
# -----------------------------
shopping_bot = Agent(
    role="Shopping Specialist for folding ebikes",
    goal="Extract specifications and attributes of ebike models from e-commerce sites.",
    backstory="""A helpful shopping assistant who is an expert in product search and 
    comparing specifications and prices for consumer goods. You specialize in folding bikes.""",
    tools=[shopping_web_search_tool],
    llm=llm,
    allow_delegation=False,
    verbose=True,
    max_iter=15
)

shopping_task_1 = Task(
    description="""Find technical information on the following electric folding bikes:
    - Tern Vektron, GSD
    - BTWIN E-Fold 500, E-Fold 900
    - Brompton Electric C Line, Electric P Line
    - DAHON K-Feather, Unio E20, K-ONE
    - Jimove MC Pro 2.0, MC Pro 3.0

    Do a factual extraction of their technical specifications. Include:
    1. An image of the bike
    2. Wheel size
    3. Number of gears
    4. Weight (in kg)
    5. Type of brakes
    6. Folded height (in cm)
    7. Folded length (in cm)
    8. E-system
      (a) Motor
      (b) Battery
      (c) Range
      (d) Charger

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
    output_file="output_files/specs_data.json"
)

shopping_task_2 = Task(
    description="""Find technical information on the electric folding bikes from 
    each of the competitors {competitors} mentioned in the market research task. 
    For each competitor, identify their flagship folding ebike model and extract 
    the technical specifications of these models.

    Include:
    1. An image of the bike
    2. Wheel size
    3. Number of gears
    4. Weight (in kg)
    5. Type of brakes
    6. Folded height (in cm)
    7. Folded length (in cm)
    8. E-system
      (a) Motor
      (b) Battery
      (c) Range
      (d) Charger

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
    output_file="output_files/specs_data.json"
)

specs_analyst = Agent(
    role="Senior Sales Engineer",
    goal="Analyze and compare technical specifications of competitor products to form insights.",
    backstory="""An experienced sales engineer with expertise in creating technical proposals 
    and delivering presentations to help customers understand how to choose folding bikes 
    including power-assisted ones.""",
    tools=[code_interpreter, file_writer_tool],
    #allow_code_execution=True, # need Docker-in-Docker
    llm=llm,
    allow_delegation=False,
    verbose=True,
    max_iter=5
)

visualize_specs_task = Task(
    description="""                                                                 │
│   1. Locate the JSON file at 'output_files/specs_data.json'.
    2. Generate visualization for the JSON data and output interactive HTML file. 
    The visualizations should break down the ebike specifications to help the user compare 
    them based on product attributes such as weight, range, and battery capacity.
    3. Discuss how the different ebike models differentiate, and provide insights for consumer 
    decision making.
    """,
    expected_output="""A well-structured and insightful comparison of the ebike specifications 
    from the JSON data, illustrated by high-quality charts.""",
    context=[shopping_task_1],
    agent=specs_analyst,
)

# -----------------------------
# Input guardrail
# -----------------------------
topic_guard_agent = Agent(
    role='Topic Guardrail Agent',
    goal="""Ensure user questions are related to {topic}.""",
    backstory="""A security expert specialized in ensuring that conversations
    stay on-topic and tasks are within scope. If a question is off-topic, you
    terminate the conversation and stop the crew.""",
    tools=[obj_detector_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

check_topic_task = Task(
    description="""Analyze the user inputs: {product} and {new_color}.
    Determine if {product} is about {topic} AND {new_color} is a valid color.
    Return 'ALLOWED' or 'OFF_TOPIC'.""",
    expected_output="ALLOWED or OFF_TOPIC",
    agent=topic_guard_agent
)

check_input_image_task = Task(
    description="""Use tool to detect if the image at {image_url} contains a bicycle. 
    Return 'ALLOWED' or 'OFF_TOPIC'.""",
    expected_output="ALLOWED or OFF_TOPIC",
    agent=topic_guard_agent
)

aggregate_checks_task = Task(
    description="If any check is OFF_TOPIC, return OFF_TOPIC. Else ALLOWED.",
    context=[check_topic_task, check_input_image_task],
    expected_output="Final status string",
    agent=topic_guard_agent
)

# -----------------------------
# Flow state and class
# -----------------------------

class MarketResearchState(BaseModel):
    query_type: str = ""
    product: str = ""
    video_url: str = ""
    image_url: str = ""
    new_color: str = ""
    analysis: CompetitiveAnalysis | None = None

class MarketResearchFlow(Flow[MarketResearchState]):
    """Flow for performing market research for consumer product"""

    @start()
    def start_flow(self):
        """Initialize"""
        print(f"Starting flow for {self.state.query_type}")

        # Ensure output directory exists before saving
        os.makedirs("output_files", exist_ok=True)

        return {"query_type": self.state.query_type,
                "product": self.state.product,
                "video_url": self.state.video_url,
                "image_url": self.state.image_url,
                "new_color": self.state.new_color
                }

    @router(start_flow)
    def route_query(self):
        query_type = self.state.query_type
        if query_type == "Market research":
            return "research_crew"
        elif query_type == "Variant generation":
            return "image_crew"
        elif query_type == "Video transcription":
            return "video_crew"
        elif query_type == "Specs data collection":
            return "specs_crew"
        return "unsupported"
    
    @listen("research_crew")
    async def analyze_market(self):
        """Conduct market research on product"""
        st.divider()
        #st.write(f"Starting market research for {self.state.product}") 

        research_crew = Crew(
            agents=[reddit_researcher, analyst, writer, editor],
            tasks=[reddit_search_task, visualize_sentiments_task, market_research_task, writing_task, editing_task],
            process=Process.sequential, # Process.sequential | Process.hierarchical
            #manager_llm=llm, # manager_llm=llm | manager_agent=manager
            planning=True,
            memory=True,
            verbose=True,
            output_log_file="output_files/research_crew_log"
        )

        crew_inputs = {"product": self.state.product}

        with st.spinner("Performing market research..."):
            result = await research_crew.kickoff_async(inputs=crew_inputs)

        st.markdown("### ✨ Results:")
        if result.pydantic:
            st.write(result.pydantic)
        else:
            st.write(result.raw)

        # Return the analysis to update the state
        return {"analysis": result.pydantic}
    
    @listen("video_crew")
    async def transcribe_video(self):
        """Summarize YouTube video"""
        st.divider()        
        #st.write(f"Starting summary of YouTube video at {self.state.video_url}")

        video_crew = Crew(
            agents=[video_researcher],
            tasks=[video_research_task],
            verbose=True,
            output_log_file="output_files/video_crew_log"
        )

        crew_inputs = {
            "video_url": self.state.video_url
        }

        with st.spinner("Analyzing YouTube video..."):
            result = await video_crew.kickoff_async(inputs=crew_inputs)

        st.markdown("### ✨ Results:")
        st.write(result.raw)

    @listen("image_crew")
    async def generate_variant(self):
        """Analyze existing product image and generate a variant"""
        st.write(f"Starting variant generation for {self.state.image_url}")

        image_crew = Crew(
            agents=[image_analyst],
            #tasks=[describe_image_task, generate_variant_task],
            tasks=[generate_image_variant_task],
            process=Process.sequential,
            planning=True,
            verbose=True,
            output_log_file="output_files/image_crew_log"
        )

        crew_inputs = {
            "image_url": self.state.image_url,
            "new_color": self.state.new_color
        }

        with st.spinner("Generating image variant..."):
            result = await image_crew.kickoff_async(inputs=crew_inputs)

        st.divider()
        st.markdown("### ✨ Results:")
        save_path=f"output_files/generated_variant_{self.state.new_color}.jpg"
        download_image(result, save_path=save_path)
        if os.path.exists(save_path):
            st.image(save_path, caption=f'Product variant in {self.state.new_color}', width=400)
        else:
            st.warning("Failed to generate image variant.")

    #@listen(analyze_market)
    @listen("specs_crew")
    async def collect_specs(self, analysis):
        """Search for competitor product specs"""
        st.divider()        
        #st.write("Starting search for product specifications from competitors:")

        _ = '''
        # Extract competitor names from the market research analysis
        with open('output_files/research.json', 'r') as f:
            data = json.load(f)           
            competitor_names = [competitor['name'] for competitor in data['competitors']]

        for competitor in competitor_names:
            st.write(f"- {competitor}")
        
        crew_inputs = {"competitors": competitor_names}
        '''

        specs_crew = Crew(
            agents=[shopping_bot, specs_analyst],
            tasks=[shopping_task_1, visualize_specs_task], # shopping_task_1 | shopping_task_2
            process=Process.sequential,
            planning=True,
            memory=True, 
            verbose=True,
            output_log_file="output_files/specs_crew_log"
        )

        with st.spinner("Searching for product specifications..."):
            #result = await shopping_crew.kickoff_async(inputs=crew_inputs)
            result = await specs_crew.kickoff_async()

        st.markdown("### ✨ Results:")
        #st.write("Specs data collection saved to output_files/specs_data.json")
        st.write(result.raw)

    @listen("unsupported")
    def exit_flow(self):
        st.warning(f"Sorry, Unsupported query type.")
        return "Exiting flow"

_ = '''
    @listen(or_(analyze_market, generate_variant))
    @human_feedback(
            message="Please review this draft. Reply 'approved' or 'rejected'.",
            emit=["approved", "rejected"],
            llm=llm,
            default_outcome = "rejected"
    )
    def review_content(self):
        """Get human feedback on the generated content"""
        return "Awaiting human feedback..."

    @listen("approved")
    def on_approval(self, result: HumanFeedbackResult):
        st.write(f"Content approved! Feedback: {result.feedback}")

    @listen("rejected")
    def on_rejection(self, result: HumanFeedbackResult):
        st.write(f"Content rejected. Reason: {result.feedback}")
'''

# Define guard crew outside of flow class
guard_crew = Crew(
    agents=[topic_guard_agent],
    tasks=[check_topic_task, check_input_image_task, aggregate_checks_task],
    process=Process.sequential
)

# --- end of agentic system code ---

# Validate inputs before passing to crew
class InputValidator(BaseModel):
    query_type: str
    product: str
    video_url: str
    image_url: str
    new_color: str
    topic: str

    @field_validator('query_type', 'product', 'video_url', 'image_url', 'new_color', 'topic')
    @classmethod
    def check_not_empty(cls, v) -> str:
        if v is None or len(v.strip()) == 0:
            raise ValueError('Field cannot be null or empty')
        return v

    @field_validator('image_url')
    @classmethod
    def check_file_exists(cls, v: str) -> str:
        # Check if the file exists using pathlib
        if not Path(v).exists():
            raise ValueError(f"File not found: {v}")
        return v

# --- runtime functions ---

def clear_output_folder(folderPath="output_files"):
    if os.path.exists(folderPath):
        filesList = glob.glob(folderPath + "/*")
        for file in filesList:
            os.remove(file)

async def run_crew_async(crew: Crew, inputs: dict):
    return crew.kickoff(inputs=inputs)

async def run_flow_async():
    flow = MarketResearchFlow()
    flow.plot(filename="market_research_flow.html") # file saved to /tmp/crewai_flow_i*
    result = await flow.kickoff_async(inputs=validated_data.model_dump())
    return result

# --- PAGE CONFIG ---
icon_img = Image.open("bicycle_icon.png")
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 10])
with col1:
    st.image(icon_img, width=50)
with col2:
    st.title("AI-powered Market Research Assistant")
    #st.markdown("<h2 style='margin-top: 0px;'>AI-powered Market Research Assistant</h2>", unsafe_allow_html=True)
st.markdown("Hi, enter a folding bike product, and let me help you with the market research.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ User inputs")
    query_type = st.radio("Select task:", ['Market research', 'Variant generation', 
                            'Video transcription', 'Specs data collection'], index=0)
    product = st.text_input("Product:", placeholder="e.g., Tern folding bike", value="Tern folding bike")
    video_url = st.text_input("Video link for analysis:", value="https://www.youtube.com/watch?v=lhDoB9rGbGQ")
 
    folderPath = os.path.abspath('input_files')
    filesList = glob.glob(folderPath + "/*")
    basenames = [os.path.basename(f) for f in filesList]
    selected_image = st.selectbox("Select product image for color variation:", options=basenames, index=0)
    image_url = os.path.join(folderPath, selected_image)
    if image_url:
        st.sidebar.image(image_url, caption='Product original image', width=200)

    new_color = st.text_input("New color for product variant:", placeholder="e.g., white, blue, gold, red, green", value="white")
    st.divider()
    #st.info("Version v0.2.0")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()
        clear_output_folder("output_files")

# Validate inputs before passing to crew
raw_data = {
    "query_type": query_type,
    "product": product,
    "video_url": video_url,
    "image_url": image_url,
    "new_color": new_color,
    "topic": "Folding bicycles" # folding bikes, electric bikes
}

try:
    validated_data = InputValidator(**raw_data)
except ValidationError as e:
    st.warning(f"Validation Error: {e}")

if st.button("Run Task"):
    clear_output_folder("output_files")
    # Run guardrail crew first to check if inputs are on-topic and valid
    with st.spinner("Input guardrails..."):                    
        #result = guard_crew.kickoff(inputs=validated_data.model_dump())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_crew_async(guard_crew, inputs=validated_data.model_dump()))

    # Check for termination condition
    if "OFF_TOPIC" in str(result):
        st.warning("Please check your inputs are valid.")
    else:
        # Proceed with main agents
        asyncio.run(run_flow_async())
        st.success("✅ Flow completed!")