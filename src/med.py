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

# Load environment variables
load_dotenv()

# Configure paths and API keys
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "db")
persistent_directory = os.path.join(db_dir, "chroma_db")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("API key is missing. Please set GEMINI_API_KEY in your .env file.")

# Initialize components
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize vector store
db = None

@app.on_event("startup")
async def startup_event():
    """Initialize the vector store when the application starts"""
    global db
    db = initialize_vector_store()

# Define the directory containing your documents
file_path = os.path.join(current_dir, "..", "documents", "medical_data.txt")

def initialize_vector_store():
    """Initialize or load the vector store with medical knowledge"""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        os.makedirs(db_dir, exist_ok=True)
        print("Initializing medical knowledge base...")
        
        try:
            print("Attempting to load existing vector store...")
            client = chromadb.PersistentClient(path=persistent_directory)
            db = Chroma(
                client=client,
                embedding_function=embeddings,
                collection_name="medical_data"
            )
            print("Successfully loaded existing medical database")
            return db
        except Exception as e:
            print(f"Creating new medical database: {e}")
            
            loader = TextLoader(file_path)
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            docs = text_splitter.split_documents(documents)
            
            client = chromadb.PersistentClient(path=persistent_directory)
            
            db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persistent_directory,
                client=client,
                collection_name="medical_data"
            )
            return db
            
    except Exception as e:
        print(f"Error during medical database initialization: {e}")
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
    # Emergency related keywords
    emergency_keywords = ['emergency', 'urgent', 'severe pain', 'chest pain', 'unconscious', 'accident']
    
    # Quick remedy keywords
    quick_remedy_keywords = ['quick', 'immediate', 'remedy', 'relief', 'help now']
    
    # Detailed plan keywords
    detailed_plan_keywords = ['detailed', 'complete', 'full', 'comprehensive', 'thorough']
    
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in emergency_keywords):
        return "emergency"
    elif any(keyword in query_lower for keyword in quick_remedy_keywords):
        return "quick_remedy"
    elif any(keyword in query_lower for keyword in detailed_plan_keywords):
        return "detailed_plan"
    else:
        return "general_query"

def analyze_symptoms(query: str) -> dict:
    """Enhanced symptom analysis with disease pattern matching"""
    medical_info = {
        "symptoms": [],
        "duration": None,
        "severity": None,
        "age_group": None,
        "location": None,
        "pre_existing_conditions": None,
        "urgency_level": "normal",
        "possible_conditions": [],
        "temporal_info": {}
    }
    
    # Enhanced symptom patterns with synonyms and variations
    symptom_patterns = {
        "fever": ["fever", "high temperature", "feeling hot", "chills", "temperature"],
        "headache": ["headache", "head pain", "head ache", "migraine"],
        "stomach_pain": ["stomach pain", "abdominal pain", "belly pain", "stomach ache"],
        "rash": ["rash", "skin eruption", "spots", "skin spots", "marks on skin"],
        "fatigue": ["fatigue", "tired", "exhausted", "weakness", "lethargy"],
        "nausea": ["nausea", "feeling sick", "wanting to vomit", "queasy"],
        "diarrhea": ["diarrhea", "loose stools", "watery stools", "frequent stools"],
        "constipation": ["constipation", "hard stools", "difficulty passing stools"],
        "cough": ["cough", "coughing", "throat irritation"],
        "breathing_difficulty": ["breathing difficulty", "shortness of breath", "breathless"]
    }
    
    # Duration extraction
    duration_patterns = {
        "days": ["day", "days"],
        "weeks": ["week", "weeks"],
        "months": ["month", "months"],
        "years": ["year", "years"]
    }
    
    query_lower = query.lower()
    
    # Extract symptoms with context
    for symptom_type, patterns in symptom_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                # Extract surrounding context
                words = query_lower.split()
                pattern_index = query_lower.find(pattern)
                context_start = max(0, pattern_index - 30)
                context_end = min(len(query_lower), pattern_index + 30)
                context = query_lower[context_start:context_end]
                
                symptom_info = {
                    "type": symptom_type,
                    "raw_text": pattern,
                    "context": context
                }
                medical_info["symptoms"].append(symptom_info)
    
    # Disease pattern matching
    disease_patterns = {
        "typhoid": {
            "required": ["fever"],
            "optional": ["headache", "stomach_pain", "rash", "constipation"],
            "duration": "days",
            "min_symptoms": 3
        },
        "flu": {
            "required": ["fever"],
            "optional": ["headache", "fatigue", "cough"],
            "duration": "days",
            "min_symptoms": 2
        },
        "food_poisoning": {
            "required": ["nausea"],
            "optional": ["stomach_pain", "diarrhea", "fever"],
            "duration": "days",
            "min_symptoms": 2
        }
    }
    
    # Match symptoms against disease patterns
    present_symptom_types = set(s["type"] for s in medical_info["symptoms"])
    
    for disease, pattern in disease_patterns.items():
        required_match = all(sym in present_symptom_types for sym in pattern["required"])
        optional_count = sum(1 for sym in pattern["optional"] if sym in present_symptom_types)
        total_matches = len(pattern["required"]) + optional_count
        
        if required_match and total_matches >= pattern["min_symptoms"]:
            medical_info["possible_conditions"].append({
                "condition": disease,
                "confidence": total_matches / (len(pattern["required"]) + len(pattern["optional"]))
            })
    
    # Severity analysis with context
    severity_patterns = {
        "mild": ["mild", "slight", "minor", "little"],
        "moderate": ["moderate", "medium", "somewhat"],
        "severe": ["severe", "intense", "extreme", "unbearable", "high", "serious"]
    }
    
    # Check severity for each symptom
    for symptom in medical_info["symptoms"]:
        context = symptom["context"]
        for severity, indicators in severity_patterns.items():
            if any(indicator in context for indicator in indicators):
                symptom["severity"] = severity
                break
    
    # Overall severity assessment
    severity_scores = {"mild": 1, "moderate": 2, "severe": 3}
    severity_count = {"mild": 0, "moderate": 0, "severe": 0}
    
    for symptom in medical_info["symptoms"]:
        if "severity" in symptom:
            severity_count[symptom["severity"]] += 1
    
    if severity_count["severe"] > 0:
        medical_info["severity"] = "severe"
    elif severity_count["moderate"] > severity_count["mild"]:
        medical_info["severity"] = "moderate"
    else:
        medical_info["severity"] = "mild"
    
    return medical_info

def get_medical_prompt(medical_info: dict, intent: str) -> str:
    """Generate friendly, action-oriented medical advice prompt"""
    # Format conditions with friendly tone
    conditions_text = ""
    if medical_info["possible_conditions"]:
        conditions = sorted(medical_info["possible_conditions"], 
                          key=lambda x: x["confidence"], 
                          reverse=True)
        conditions_text = "Based on your symptoms, this could be:\n"
        for cond in conditions:
            conditions_text += "- {}\n".format(cond['condition'].title())

    # Check if condition is infectious
    infectious_conditions = ["typhoid", "flu", "tuberculosis", "covid", "chickenpox"]
    is_infectious = any(cond["condition"] in infectious_conditions 
                       for cond in medical_info.get("possible_conditions", []))

    # Format immediate actions section
    immediate_actions = """
    IMMEDIATE STEPS TO TAKE:
    1. Self-Care Actions (what you can do right now)
    2. Quick Home Remedies (using common household items)
    3. Medication Guidance (if any over-the-counter medicines can help)
    4. When to See a Doctor
    """

    if is_infectious:
        immediate_actions += "5. IMPORTANT: Isolation and family protection measures\n"

    if medical_info["urgency_level"] == "emergency":
        return """
        I understand you're in a concerning situation. Let's act quickly to help you.
        
        {}
        
        URGENT ACTIONS NEEDED:
        1. 🚨 First Steps to Take Right Now:
           - Specific immediate actions
           - Available home remedies
           - What NOT to do
        
        2. 🏥 Medical Help:
           - When and where to get emergency care
           - What to tell the medical team
        
        3. ⚡ While Waiting for Help:
           - Basic first aid steps
           - Position and comfort measures
           - Warning signs to watch for
        
        I'll guide you through each step clearly. Your safety is the priority.
        """.format(conditions_text)
    
    if intent == "quick_remedy":
        return """
        I hear you're not feeling well. Let me help you feel better quickly.
        
        {}
        
        {}
        
        Let me provide simple, effective solutions you can try right away using things you likely have at home.
        I'll also make it clear when you should seek medical help if things don't improve.
        
        Remember: Your health is important, and these are initial support measures. 
        {}
        """.format(
            conditions_text,
            immediate_actions,
            'Since this might be infectious, I\'ll also tell you how to protect your family.' if is_infectious else ''
        )
    
    return """
    I understand you'd like a detailed look at your health situation. Let me help you understand what's happening and what you can do.
    
    {}
    
    {}
    
    Let me break this down into clear, actionable steps:

    1. 🏥 Understanding Your Condition:
       - What might be causing this
       - How it typically develops
       - What factors might be making it worse

    2. 🏠 Your Care Plan:
       - Immediate relief measures
       - Step-by-step home care guide
       - Lifestyle adjustments that can help

    3. 💪 Treatment Approach:
       - Effective home remedies
       - Traditional medicine options
       - When to consider medical help

    4. ⚠️ Stay Alert For:
       - Signs of improvement
       - Warning signals
       - When to contact a doctor

    5. 🌟 Taking Care Going Forward:
       - Foods that can help
       - Daily habits to consider
       - How to prevent this in future
    
    {}
    
    I'm here to guide you through each step and help you feel better soon.
    """.format(
        conditions_text,
        immediate_actions,
        'Since this condition can spread, I\'ll also include important steps to protect your family members.' if is_infectious else ''
    )

def get_personalized_response(query: str, db: Chroma) -> str:
    """Generate enhanced personalized health guidance"""
    try:
        # Analyze query intent and enhanced symptoms
        intent = analyze_query_intent(query)
        medical_info = analyze_symptoms(query)
        
        # Get relevant medical knowledge with compatible retrieval parameters
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,  # Number of documents to retrieve
                "filter": None,  # No filtering
            }
        )
        
        # Use invoke() instead of get_relevant_documents()
        docs = retriever.invoke(query)
        
        # Generate appropriate prompt
        prompt = get_medical_prompt(medical_info, intent)
        
        # Combine with retrieved knowledge
        retrieved_context = "\n".join([doc.page_content for doc in docs])
        
        # Create focused prompt based on detected conditions
        condition_specific_prompts = ""
        for condition in medical_info["possible_conditions"]:
            condition_specific_prompts += f"\nFor {condition['condition']}, focus on specific symptoms, progression, and traditional remedies known to help."
        
        full_prompt = f"""
        {prompt}
        
        Additional Context:
        {condition_specific_prompts}
        
        Retrieved Medical Knowledge:
        {retrieved_context}
        
        Provide a clear, practical response focused on immediate help and traditional remedies available in rural settings.
        If this might be {[c['condition'] for c in medical_info['possible_conditions']]}, include specific guidance for managing these conditions.
        """
        
        return generate_gemini_response(full_prompt)
    
    except Exception as e:
        print(f"Error in get_personalized_response: {e}")
        return "Error generating response"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """API endpoint for health assistance"""
    try:
        response = get_personalized_response(request.query, db)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)