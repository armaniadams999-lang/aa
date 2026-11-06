import os
import pandas as pd
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from google.genai.errors import APIError
from gtts import gTTS

# Initialize FastAPI app
app = FastAPI(title="AI Vocal Data Agent API", version="1.0")

# Securely fetch API key from environment variables (e.g., Render/Railway environment secrets)
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable is not set.")
    client = None
else:
    client = genai.Client(api_key=api_key)

# --- Exception Definitions ---
class AgentError(Exception):
    """Base exception for our AI Data Agent."""
    pass

class AgentFileError(AgentError):
    """Exception raised when file reading or directory scanning fails."""
    pass

class AgentAPIError(AgentError):
    """Exception raised when the generative AI model API fails or times out."""
    pass

# --- Agent Classes Lineage ---
class FixedAIDataAgent:
    def __init__(self, client):
        self.client = client
        # Use a relative or configurable path for deployment (e.g., local sample_data folder)
        self.data_directory = "./sample_data/"

    def get_dataset_summary(self):
        """Automatically reads and summarizes available CSV files."""
        summaries = {}
        if not os.path.exists(self.data_directory):
            return "Data directory ./sample_data/ not found."
        for file in os.listdir(self.data_directory):
            if file.endswith('.csv'):
                path = os.path.join(self.data_directory, file)
                try:
                    df = pd.read_csv(path, nrows=5)
                    summaries[file] = {
                        "columns": list(df.columns),
                        "preview": df.to_dict(orient='records')
                    }
                except Exception as e:
                    summaries[file] = f"Error reading file: {e}"
        return summaries

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        retry=retry_if_exception_type(APIError),
        reraise=True
    )
    def ask_with_retry(self, user_query):
        if not self.client:
            raise ValueError("GenAI Client was not properly initialized. Check your GOOGLE_API_KEY environment variable.")

        dataset_context = self.get_dataset_summary()
        system_instruction = (
            "You are an expert AI Data Scientist Agent. You have access to local datasets. "
            "Use the metadata context provided to answer the user's questions about the datasets "
            "accurately and suggest analysis steps."
        )
        prompt = f"""
        Available Dataset Metadata:
        {dataset_context}

        User Question:
        {user_query}
        """
        response = self.client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=dict(system_instruction=system_instruction)
        )
        return response.text

    def __getattr__(self, name):
        """Intercepts missing attributes or methods and fallbacks to ask_with_retry."""
        print(f"[Fallback Interceptor] '{name}' method called. Routing query to 'ask_with_retry'...\n")
        return self.ask_with_retry

class AdvancedAIDataAgent(FixedAIDataAgent):
    def get_dataset_summary(self):
        """Automatically reads datasets with specialized file error handling."""
        summaries = {}
        if not os.path.exists(self.data_directory):
            raise AgentFileError(f"Target data directory '{self.data_directory}' does not exist.")

        for file in os.listdir(self.data_directory):
            if file.endswith('.csv'):
                path = os.path.join(self.data_directory, file)
                try:
                    df = pd.read_csv(path, nrows=5)
                    summaries[file] = {
                        "columns": list(df.columns),
                        "preview": df.to_dict(orient='records')
                    }
                except Exception as e:
                    raise AgentFileError(f"Failed to process CSV file '{file}': {e}") from e
        return summaries

    def ask_with_robust_handling(self, user_query):
        """Executes query and translates exceptions into custom domain exceptions."""
        try:
            dataset_context = self.get_dataset_summary()
        except AgentFileError as fe:
            print(f"[File Error Handled]: {fe}")
            raise

        try:
            return self.ask_with_retry(user_query)
        except Exception as ae:
            raise AgentAPIError(f"Gemini API request failed: {ae}") from ae

class VocalAIDataAgent(AdvancedAIDataAgent):
    def __init__(self, client, lang='en'):
        super().__init__(client)
        self.lang = lang

    def speak_reply(self, text_content):
        """Converts the text response to speech and saves it to a file."""
        if not text_content:
            return None

        clean_text = text_content.replace('**', '').replace('###', '').replace('##', '').replace('*', '')

        try:
            tts = gTTS(text=clean_text, lang=self.lang, slow=False)
            temp_filename = "agent_reply.mp3"
            tts.save(temp_filename)
            return temp_filename
        except Exception as e:
            print(f"Failed to generate speech using gTTS: {e}")
            return None

# Instantiate global agent if client is ready
agent = VocalAIDataAgent(client) if client else None

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response_text: str
    audio_file: str | None = None

# --- FastAPI Endpoints ---
@app.post("/ask", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="GenAI client not initialized. Check server environment variables.")
    
    try:
        response_text = agent.ask_with_robust_handling(request.query)
        audio_file = agent.speak_reply(response_text)
        return QueryResponse(response_text=response_text, audio_file=audio_file)
    except AgentError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "running", "model": "gemini-3.8-flash"}
