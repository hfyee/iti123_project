'''
Docstring for agentic_rag_crewai_v2
'''
# Install dependencies
#!pip install crewai crewai_tools langchain_community langchain_pinecone langchain_huggingface sentence-transformers langchain-tavily langchain_openai gdown chainlit
#!pip install wikipedia youtube_search

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["USER_AGENT"] = "iti123_project/1.0 (xxxx122D@myaccount.nyp.edu.sg)"

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

'''
RAG v1: RagTool with ChromaDB as vector store

vectordb: VectorDbConfig = {
    "provider": "chromadb", # alternative: qdrant
    "config": {
        "collection_name": "ebike-docs",
        "persist_directory": "./ebike-docs_db"
    }
}

# default embedding model is openai/text-embedding-3-large
embedding_model: ProviderSpec = {
    "provider": "openai",
    "config": {
        "model_name": "text-embedding-3-small"
    }
}

config: RagToolConfig = {
    "vectordb": vectordb,
    "embedding_model": embedding_model,
    "top_k": 4 # default is 4
}

rag_tool = RagTool(
    name="Documentation Tool",
    description="Use this tool to answer questions about the knowledge base.",
    config=config,
    summarize=True
)

# Add content from PDF
rag_tool.add(data_type="pdf_file", path="https://onemotoring.lta.gov.sg/content/dam/onemotoring/Buying/PDF/PAB/List_of_Approved_PAB_Models.pdf")
# Add content from web page
rag_tool.add(data_type="website", url="https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html")
'''

# RAG v3: Pinecone vector store with reranking

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
pdf_loader = create_directory_loader('.pdf', './rag_docs')
txt_loader = create_directory_loader('.txt', './rag_docs')

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
    "https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html",
    "https://www.consumerreports.org/health/bikes/best-folding-bikes-a2576871382/"
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
#all_docs = txt_docs + pdf_docs

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
    name: str = "tavily_search"
    description: str = "A tool for searching the internet using Tavily to get real-time information."
    args_schema: Type[BaseModel] = TavilySearchInput
    search: TavilySearch = Field(default_factory=TavilySearch)

    def _run(self, query: str, search_depth: str, include_domains: str, include_images: bool, include_image_descriptions: bool) -> str:
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

generation = GenerationTool()

web_search = TavilySearchTool(
    search_depth="basic",
    include_images = True,
    include_image_descriptions = True,
    include_domains = [
        "https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/vehicle-types-and-registrations/PAB.html",
        "https://www.brompton.com/stories/design-and-engineering",
        "https://dahon.com/technology"
    ])

wiki = WikipediaTool()
dalle = DallEImageTool()
youtube = YouTubeSearchTool()
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

'''
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

# Define the agents
# Enable coordination by setting allow_delegation=True for the manager agent
# Manager agent is not allowed to have tools.
crew_manager = Agent(
    role='Crew Manager',
    goal='Efficiently manage the crew and ensure high-quality task completion',
    backstory=(
        """With years of experience leading a team, you are an expert at interpreting user questions and delegating tasks to coworkers.
        You're methodical in your approach to coordinating the workflow and reviewing results from coworkers.
        """
    ),
    allow_delegation=True,
    verbose=True,
    max_iter=5, # prevent infinite loops, default is 25
    llm=llm,
)

knowledge_expert = Agent(
    role='Customer Support',
    goal='Uses RAG tool to answer questions about the knowledge base.',
    backstory=(
        """You have a talent for fetching information from knowledge base."""
    ),
    tools=[rag_tool],
    allow_delegation=False,
    verbose=True,
    max_iter=5, # prevent infinite loops, default is 25
    llm=llm,
)

researcher = Agent(
    role='Market Researcher',
    goal='Conduct market research about the assigned consumer product that provides valuable insights to the reader.',
    backstory=(
        """You have years of experience conducting market research for leading consumer goods companies.
        You have a gift for analysis, reading between the lines and identifying patterns that others miss.
        You excel at explaining complex concepts in accessible language."""
    ),
    allow_delegation=False,
    verbose=True,
    max_iter=5, # prevent infinite loops, default is 25
    llm=llm,
)

content_reviewer = Agent(
    role="Content Reviewer and Editor",
    goal='Ensure content is accurate, comprehensive, well-structured, and insightful with clear takeaways.',
    backstory=(    
        """You are a meticulous editor with years of experience reviewing market reports from consultants. 
        You have an eye for clarity and coherence. You excel at improving content, while maintaining the original author's voice 
        and ensuring consistent quality across multiple sections in the report."""
    ),
    allow_delegation=False,
    verbose=True,
    max_iter=5, # prevent infinite loops, default is 25
    llm=llm,
)

# Define the tasks
# 80/20 rule: focus on detailed tasks over agents
'''
manage_task = Task(
    description=(
        """Analyse {question} and delegate tasks to coworkers."""
    ),
    human_input=True, # The agent can prompt the user for input
    expected_output='One word: support, researcher, writer or finished.',
    agent=crew_manager,
)
'''
research_task = Task(
    description=(
        """ Create a market report on the consumer good market mentioned in {question} for Singapore market.

            Target audience: product strategy team

            The report should include: 
            1. Key trends
            2. The top 2 players in this space, their positioning, reach, and ratings
            3. Potential market gaps and any emerging players
            4. Relevant product image or video URLs
            5. Be approximately 500-800 words in length
        """),
    expected_output=(
        'A well-structured, comprehensive report in Markdown format that is informative and appropriate for the target audience.'
    ),
    tools=[web_search, wiki, youtube, file_writer],
    human_input=False, # the agent can prompt the user for input
    output_file='./output/research.md',
    agent=researcher
)

review_task = Task(
    description=(
        """Review and improve the market report from the Researcher.

        Target audience: product strategy team

        Your review should:
        1. Fix any grammatical or spelling errors
        2. Improve clarity and readability
        3. Ensure content is comprehensive and accurate
        4. Enhance the structure and flow
  
        Provide the improved version of the report in Markdown format.
        """),
    expected_output=(
        'An improved, polished version of the report that maintains the original structure but enhances clarity, accuracy and consistency.'
    ),
    tools=[file_writer],
    human_input=True, # the agent can prompt the user for input
    output_file='./output/reviewed.md',
    agent=content_reviewer
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
    memory=True,
    verbose=True
)

pd_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.hierarchical,
    manager_llm=llm,
    planning=False,
    memory=True,
    verbose=True
)
'''

# Run the crews within Chainlit
@cl.on_chat_start
async def on_chat_start():
    # Store the crew object in the user session
    crew = Crew(
        agents=[knowledge_expert, researcher, content_reviewer],
        tasks=[research_task, review_task],
        process=Process.hierarchical,
        manager_llm=llm,
        #manager_agent=crew_manager,
        planning=False,
        memory=True,
        verbose=True
    )
    
    cl.user_session.set("my_crew", crew)
    #await cl.Message(content="CrewAI system ready. How can I help you?").send()
    await cl.Message(content="CrewAI system ready. The consumer good category to research:").send()

@cl.on_message
async def on_message(message: cl.Message):
    crew = cl.user_session.get("my_crew")

    # In a conversational loop, you might dynamically create a task or have a standing 'conversation task'
    # The basic execution method takes an input string
    # The agent will handle subsequent human inputs via the HumanInTheLoopTool internally
    try:
        #result = crew.kickoff(inputs={"question": message.content})
        result = await asyncio.to_thread(lambda: crew.kickoff({'question': message.content}))  
        await cl.Message(content=result).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()
