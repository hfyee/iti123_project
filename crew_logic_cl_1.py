'''
Docstring for agentic_rag_crewai_v2
'''
# Install dependencies
#!pip install crewai crewai_tools langchain_community langchain_pinecone langchain_huggingface sentence-transformers langchain-tavily langchain_openai gdown chainlit
#!pip install wikipedia youtube_search
'''
# Download env file or use local copy
import gdown
url = 'https://drive.google.com/file/d/17C0MsdQ0gN9bHML_dYOQQ1CUxzIdkF0q/view?usp=drive_link' # HF
output_path = '.env'
gdown.download(url, output_path, quiet=False,fuzzy=True)
'''

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["USER_AGENT"] = "iti123_project/1.0 (9108122D@myaccount.nyp.edu.sg)"

# Import crewai packages
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool
from crewai_tools import WebsiteSearchTool
from crewai_tools import RagTool
from crewai_tools.tools.rag import RagToolConfig, VectorDbConfig, ProviderSpec
from crewai_tools import FileWriterTool
from pydantic import BaseModel, Field
from typing import Type

# Import langchain packages
from langchain_openai import ChatOpenAI
# For RAG pipeline
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
# For agent tools
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

# Import chainlit packages
import chainlit as cl
from chainlit import run_sync
import asyncio

# Load the LLM
llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-4o-mini",
    temperature=0.2, # lower temp for focused management
    max_tokens=1000,
)

# Start of RAG pipeline

# RAG document loaders
# Define a dictionary to map file extensions to their respective loaders
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
pdf_loader = create_directory_loader('.pdf', './docs2upload')
txt_loader = create_directory_loader('.txt', './docs2upload')

# Load the documents
pdf_docs = pdf_loader.load()
txt_docs = txt_loader.load()

print(f"Loaded {len(pdf_docs)} PDF files.")
for doc in pdf_docs:
    print(f"Metadata:\n{doc.metadata}\n")
    print(f"Content snippet:\n{doc.page_content[:100]}...\n")

print(f"Loaded {len(txt_docs)} text files.")
for doc in txt_docs:
    print(f"Metadata:\n{doc.metadata}\n")
    print(f"Content snippet:\n{doc.page_content[:100]}...\n")

# Website loader
urls = [
    "https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html"
]

web_loader = WebBaseLoader(urls)

# Load the documents
web_docs = web_loader.load()

print(f"Loaded {len(web_docs)} URLs.")
for doc in web_docs:
    print(f"Metadata:\n{doc.metadata}\n")
    print(f"Content snippet:\n{doc.page_content[:100]}...\n")

# RAG text splitter
# Split documents into chunks
# chunk_overlap ~10% of chunk_size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

all_docs = web_docs + txt_docs + pdf_docs

split_docs = text_splitter.split_documents(all_docs)

# RAG with Pinecone as vector store and reranking
# Create RAG tool with custom configuration
class PineconeRerankTool(BaseTool):
    name: str = "Pinecone Advanced RAG Search"
    description: str = (
        "Useful for retrieving specific information from the internal knowledge base "
        "stored in Pinecone with reranking. Use this to find documents, context, or past data."
    )

    index_name: str = "iti123-openai-index"

    def _run(self, query: str) -> str:
        # Initialize embeddings (must match what you used to ingest data)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv('OPENAI_API_KEY'), dimensions=512)

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

rag_tool = PineconeRerankTool(index_name="iti123-openai-index")

# End of RAG pipeline

# Define Agent Tools
class GenerationTool(BaseTool):
    name: str = 'Generation'
    description: str = 'Useful for general queries answered by the LLM.'

    def _run(self, query: str) -> str:
        return llm.invoke(query)

# Define input schema for the tool
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool"""
    query: str = Field(description="The search query to look up on the internet.")
    search_depth: str = Field(default="basic", description="The depth of the search results to return.")
    include_domains: str = Field(default="", description="A comma-separated list of domains to include in the search results.")
    include_images: bool = Field(default=False, description="Whether to include images in the search results.")
    include_image_descriptions: bool = Field(default=False, description="Whether to include image descriptions in the search results.")

class TavilySearchTool(BaseTool):
    name: str = "tavily"
    description: str = "A tool for searching the internet using Tavily to get real-time information."
    args_schema: Type[BaseModel] = TavilySearchInput # Link the schema to the tool
    search: TavilySearch = Field(default_factory=TavilySearch)

    def _run(
        self,
        query: str,
        search_depth: str = "basic", # basic|advanced
        include_domains: str = [
            "https://www.brompton.com/stories/design-and-engineering", 
            "https://dahon.com/technology",
            "https://ekolife.asia/",
            "https://www.consumerreports.org/health/bikes/best-folding-bikes-a2576871382/"
        ],
        include_images: bool = False,
        include_image_descriptions: bool = False,
        ) -> str:
        return self.search.run(query)

class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = "A tool to search for topics on Wikipedia and return a summary of the article."
    #search: WikipediaAPIWrapper = Field(default_factory=WikipediaAPIWrapper)

    def _run(self, query: str) -> str:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
        #return self.search.run(query)
        return api_wrapper.run(query)

class DallEImageTool(BaseTool):
    name: str = "dalle"
    description: str = "Useful for when you need to generate an image from a text prompt."

    def _run(self, query: str) -> str:
        api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1792x1024")
        return api_wrapper.run(query)

class YouTubeSearchTool(BaseTool):
    name: str = "youtube"
    description: str = "Useful for when you need to search for videos on YouTube."
    search: YouTubeSearchTool = Field(default_factory=YouTubeSearchTool)

    def _run(self, query: str) -> str:
        return self.search.run(query)

web_search = TavilySearchTool()
wiki = WikipediaTool()
dalle = DallEImageTool()
youtube = YouTubeSearchTool()
generation = GenerationTool()
file_writer = FileWriterTool()

# Define tool for human-in-the-loop interaction
class HumanInTheLoopTool(BaseTool):
    name: str = "human_in_the_loop" 
    #description: str = "A tool to ask the human user for input when the agent is unsure about how to proceed."   
    description: str = "Use this tool to ask the human user follow-up questions to get additional context."
    #description: str = "Use this tool to ask the human user to provide feedback on the final result."

    def _run(self, query: str) -> str:
        # Send message to Chainlit frontend and wait for user response
        human_response = run_sync(cl.AskUserMessage(content=query).send_and_wait_for_response())
        if human_response:
            return human_response.content
        return "No response received from the user."
    
human_in_the_loop = HumanInTheLoopTool()

# ConversationChain setup for memory management
from langchain.agents import Tool
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="The following is a friendly conversation between a human and an AI. The AI is talkative and provides adequate specific details from its context.\n\n"
             "Current conversation:\n{chat_history}\nHuman: {input}\nAI:"
)

'''
# Initialize the ConversationBufferMemory
# The memory_key should match the history variable in the prompt
memory = ConversationBufferMemory(memory_key="chat_history")

# Create the ConversationChain
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Define conversation summarization tool
class ChatSummaryTool(BaseTool):
    name: str = "chat_summary"
    description: str = "Useful for when you summarize a conversation with the human user."

    def _run(self, query: str) -> str:
        return conversation.run

chat_summary_tool = ChatSummaryTool()
'''

# Create the agents with the defined tools
# Enable coordination by setting allow_delegation=True for the manager agent
# Manager agent is not allowed to have tools.
crew_manager = Agent(
    role='Crew Manager',
    goal='Efficiently manage the crew and ensure high-quality task completion',
    backstory=(
        """With 10+ years of experience leading a team, you are an expert at interpreting user questions and delegating tasks to coworkers.
        You're methodical in your approach to coordinating the workflow and reviewing results from coworkers.
        """
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

customer_support = Agent(
    role='Customer Support',
    goal='Retrieve accurate information from the vector store to answer the question.',
    backstory=(
        """You have spent a decade in customer support for major consumer product companies.
        You are meticulous, and have a talent for understanding product datasheets and fetching information from a knowledge base."""
    ),
    tools=[rag_tool, web_search],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

researcher = Agent(
    role='Market Researcher',
    goal='Find information about the latest product trends and launches for the Asian market.',
    backstory=(
        """You have spent 15 years conducting and analyzing user research for top companies in the urban mobility space.
        You have a talent for reading between the lines and identifying patterns that others miss."""
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

writer = Agent(
    role="Technical Writer",
    goal='Create compelling content based on research that uses product features as your sales tool.',
    backstory="As a seasoned copywriter with a deep understanding of technical products, you can transform complex research into easy-to-read articles, mimicking a specific brand voice.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Define the tasks
# 80/20 rule: focus on detailed tasks over agents
'''
manage_task = Task(
    description=(
        """Analyse the question, {question}, and delegate tasks to coworkers."""
    ),
    human_input=True, # The agent can prompt the user for input
    expected_output='One word: support, researcher, writer or finished.',
    agent=crew_manager,
)
'''

support_task = Task(
    description='Find the answer to the question, {question}, in the vector store.',
    expected_output='A JSON report summarizing the {question} and the answer retrieved.',
    #tools=[rag_tool], # pdf_search, lta_website, dir_search
    human_input=True, # The agent can prompt the user for input
    output_file='cs_report.md',
    agent=customer_support
)

# The primary goal of the research phase is info gathering to build a comprehensive knowledge base about the topic.
research_task = Task(
    #description='For the product stated in the question, {question}, research the top 3 market trends.',
    description='Gather information about the folding bike technologies of the two bike makers mentioned in the question, {question}.',
    expected_output=(
        """A preliminary JSON report of raw findings, including URLs to relevant images or videos."""
    ),
    tools=[web_search, wiki, youtube],
    human_input=True, # the agent can prompt the user for input
    output_file='research.json',
    agent=researcher
)

# The primary goal of the analysis phase is interpretation and synthesis of the data collected
# during the research phase to answer the initial research questions
analysis_task = Task(
    #description='Analyze the identified trends to determine consumer needs, market gaps, and competition.',
    description='Analyze the data collected from the research phase and compare 3 technologies of the two bike makers mentioned in the question, {question}.',
    expected_output=("A final JSON report detailing the insights, conclusions, and recommendations"),
    tools=[generation, web_search],
    human_input=True, # the agent can prompt the user for input
    output_file='analysis.json',
    agent=researcher
)

writing_task = Task(
    description='Using the provided research notes, write a market research report with the product development team as the target audience.',
    expected_output=(
        """A concise Markdown report (around 500 words) that includes URLs to relevant images or videos."""
    ),
    tools=[generation, dalle, file_writer],
    human_input=True, # the agent can prompt the user for input
    output_file='mkt_report.md',
    agent=writer
)

'''
# Create the crew(s) with the defined agents and tasks
cs_crew = Crew(
    agents=[customer_support],
    tasks=[support_task],
    process=Process.hierarchical,
    manager_llm=llm,
    #manager_agent=crew_manager,
    planning=False,
    memory=True, # enable conversational memory
    verbose=True,
    output_log_file="cs_tool_usage"
)

pd_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.hierarchical,
    manager_llm=llm,
    planning=False,
    memory=True, # enable conversational memory
    verbose=True,
    output_log_file="pd_tool_usage"
)
'''
    
# Run and test the crews within Chainlit
'''
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 Hi! I'm your project assistant. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    result = await asyncio.to_thread(lambda: cs_crew.kickoff({'question': message.content}))
    await cl.Message(content=result).send()
'''

@cl.on_chat_start
async def on_chat_start():
    # Store the crew object in the user session
    cs_crew = Crew(
        agents=[customer_support],
        tasks=[support_task],
        process=Process.hierarchical,
        #manager_llm=llm,
        manager_agent=crew_manager,
        planning=False,
        memory=True, # enable conversational memory
        verbose=True,
        output_log_file="cs_tool_usage"
    )
    
    cl.user_session.set("my_crew", cs_crew)
    await cl.Message(content="CrewAI system ready. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    cs_crew = cl.user_session.get("my_crew")

    # In a conversational loop, you might dynamically create a task or have a standing 'conversation task'
    # The basic execution method takes an input string
    # The agent will handle subsequent human inputs via the HumanInTheLoopTool internally
    try:
        result = cs_crew.kickoff(inputs={"question": message.content})    
        await cl.Message(content=f"CrewAI Output: {result}").send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()