from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from typing import List, Optional
import chromadb

app = FastAPI()


# Create Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Load environment variables from .env file
load_dotenv()

# Define the persistent directory for storing the vector database
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "db")
persistent_directory = os.path.join(db_dir, "chroma_db")

# Load the API key for Google Gemini model from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("API key is missing. Please set GEMINI_API_KEY in your .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')




# Initialize vector store at startup
db = None

@app.on_event("startup")
async def startup_event():
    """Initialize the vector store when the application starts"""
    global db
    db = initialize_vector_store()


# Define the directory containing your documents
file_path = os.path.join(current_dir, "..", "documents", "data.txt")

def initialize_vector_store():
    """Initialize or load the vector store"""
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        # Create the db directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        print("Initializing vector store...")
        
        # Try to load existing database first
        try:
            print("Attempting to load existing vector store...")
            client = chromadb.PersistentClient(path=persistent_directory)
            db = Chroma(
                client=client,
                embedding_function=embeddings,
                collection_name="data"
            )
            print("Successfully loaded existing vector store")
            return db
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            print("Creating new vector store...")
            
            # If loading fails, create new database
            # Read the text content from the file
            loader = TextLoader(file_path)
            documents = loader.load()

            # Split documents into smaller chunks
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            docs = text_splitter.split_documents(documents)

            print("\n--- Creating embeddings ---")
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(path=persistent_directory)
            
            # Create the vector store
            db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persistent_directory,
                client=client,
                collection_name="data"
            )
            
            print("\n--- Vector store created and persisted ---")
            return db
        
    except Exception as e:
        print(f"Error during vector store initialization: {e}")
        raise

def generate_gemini_response(prompt: str) -> str:
    """Generate a response using the Gemini model"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response"
    

def analyze_query_intent(query: str) -> str:
    """Analyze the query to determine the primary intent"""
    # Diet plan related keywords
    diet_plan_keywords = ['diet', 'meal plan', 'food plan', 'what should i eat', 'diet chart']
    
    # Specific nutrition question keywords
    nutrition_keywords = ['benefits', 'good for', 'nutrients', 'properties', 'help with']
    
    # Location specific keywords
    location_keywords = ['in my area', 'available', 'local', 'market', 'region']
    
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in diet_plan_keywords):
        return "diet_plan"
    elif any(keyword in query_lower for keyword in nutrition_keywords):
        return "nutrition_query"
    else:
        return "general_query"

def analyze_user_inputs(query: str) -> dict:
    """Analyze if the query contains key information needed for personalized diet plan"""
    key_factors = {
        "location": None,
        "menstrual_phase": None,
        "health_conditions": None,
        "dietary_preferences": None,
        "cooking_time": None,
        "symptoms": None
    }
    
    # Location detection
    locations = ["kerala", "tamil nadu", "karnataka", "andhra", "telangana"]
    for location in locations:
        if location.lower() in query.lower():
            key_factors["location"] = location
    
    # Menstrual phase detection
    phases = {"follicular": "Days 1-14", 
             "ovulation": "Days 14-16", 
             "luteal": "Days 16-28", 
             "menstrual": "Days 1-5"}
    for phase in phases:
        if phase.lower() in query.lower():
            key_factors["menstrual_phase"] = phase
    
    # Health conditions and symptoms
    conditions = {"pcod": "PCOD", "pcos": "PCOS", "pms": "PMS", 
                 "iron deficiency": "Iron Deficiency"}
    symptoms = ["pain", "cramps", "bloating", "fatigue"]
    
    for condition in conditions:
        if condition.lower() in query.lower():
            key_factors["health_conditions"] = conditions[condition]
    
    for symptom in symptoms:
        if symptom.lower() in query.lower():
            key_factors["symptoms"] = symptom
    
    return key_factors

def get_diet_plan_prompt(key_factors: dict) -> str:
    """Generate detailed diet plan prompt using the provided structure"""
    return f"""
    You are an expert nutritionist specializing in women's health in India. Create a personalized diet plan with the following information:

    User Details:
    Location: {key_factors["location"] or "Not specified"}
    Menstrual Phase: {key_factors["menstrual_phase"] or "Not specified"}
    Health Conditions: {key_factors["health_conditions"] or "None mentioned"}
    Symptoms: {key_factors["symptoms"] or "None mentioned"}

    Please provide recommendations in exactly this structure:

    **1. Personalized Diet Overview**
    * Location-based suggestions
    * Health condition considerations
    * Phase-specific recommendations

    **2. Weekly Meal Plan ({key_factors["menstrual_phase"] or "Current"} Phase Focus)**
    * Day 1
      - Breakfast (time): [meal] (nutritional focus)
      - Lunch (time): [meal] (nutritional focus)
      - Dinner (time): [meal] (nutritional focus)
      - Snacks: [options]
    [Continue for Day 2-7]

    **3. Shopping List**
    * Essential Ingredients
    * Weekly Fresh Produce
    * Storage Tips

    **4. Health Recommendations**
    * Phase-specific focus
    * Condition management
    * Supplement needs

    **5. Practical Tips**
    * Meal prep suggestions
    * Time management
    * Storage solutions

    Keep the response practical and focused on locally available ingredients and resources.
    """

def get_general_nutrition_prompt(query: str) -> str:
    """Generate prompt for general nutrition queries"""
    return f"""
    As a nutrition expert specializing in women's health in India, please provide guidance on:
    
    Query: {query}
    
    Focus on:
    1. Direct answer to the nutrition question
    2. Scientific explanation when relevant
    3. Practical applications
    4. Local availability and alternatives if applicable
    
    Keep the response conversational and focused on the specific question.
    """

def get_personalized_response(query: str, db: Chroma) -> str:
    """Generate personalized nutrition guidance based on query analysis"""
    try:
        # Analyze user inputs
        key_factors = analyze_user_inputs(query)
        
        # Determine if this is a diet plan request
        diet_plan_keywords = ['diet plan', 'meal plan', 'diet chart', 'what should i eat', 
                            'food plan', 'nutrition plan']
        is_diet_plan_request = any(keyword in query.lower() for keyword in diet_plan_keywords)
        
        # Get relevant documents
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        docs = retriever.get_relevant_documents(query)
        
        # Choose appropriate prompt
        if is_diet_plan_request and (key_factors["location"] or 
                                   key_factors["menstrual_phase"] or 
                                   key_factors["health_conditions"]):
            prompt = get_diet_plan_prompt(key_factors)
        else:
            prompt = get_general_nutrition_prompt(query)
        
        # Combine prompt with retrieved documents
        retrieved_context = "\n".join([doc.page_content for doc in docs])
        full_prompt = """
        {prompt}

        Retrieved Context:
        {retrieved_context}
        """.format(prompt=prompt, retrieved_context=retrieved_context)

        
        
        # Generate response
        return generate_gemini_response(full_prompt)
    
    except Exception as e:
        print(f"Error in get_personalized_response: {e}")
        return "Error generating response"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """API endpoint to interact with the nutrition advisor"""
    try:
        response = get_personalized_response(request.query, db)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    API endpoint to interact with the nutrition advisor.
    """
    try:
        response = get_personalized_response(request.query, db)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# If you want to run it directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

