import os
import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from google import genai
from google.genai import types
import textwrap
from dataclasses import dataclass
from PIL import Image
import time
from ratelimit import limits, sleep_and_retry
import fitz
import io
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration class for the application"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"  # Updated to match Google's example
    TEXT_EMBEDDING_MODEL_ID: str = "text-embedding-004"  # Correct embedding model name
    DPI: int = 300  # Resolution for PDF to image conversion

class PDFProcessor:
    """Handles PDF processing using PyMuPDF and Gemini's vision capabilities"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = Config.DPI) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert PyMuPDF pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        pdf_document.close()
        return images

class GeminiClient:
    """Handles interactions with the Gemini API"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        
        # Initialize the client exactly as in Google's example
        self.client = genai.Client(api_key=api_key)
        
    def make_prompt(self, element: str) -> str:
        """Create prompt for summarization"""
        return f"""You are an agent tasked with summarizing research tables and texts from research papers for retrieval. 
                  These summaries will be embedded and used to retrieve the raw text or table elements. 
                  Give a concise summary of the tables or text that is well optimized for retrieval. 
                  Table or text: {element}"""

    def analyze_page(self, image: Image.Image) -> str:
        """Analyze a PDF page using Gemini's vision capabilities"""
        prompt = """You are an assistant tasked with summarizing images for retrieval. 
                   These summaries will be embedded and used to retrieve the raw image.
                   Give a concise summary of the image that is well optimized for retrieval.
                   If it's a table, extract all elements of the table.
                   If it's a graph, explain the findings in the graph.
                   Include details about color, proportion, and shape if necessary to describe the image.
                   Extract all text content from the page accurately.
                   Do not include any numbers that are not mentioned in the image."""
        
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=[prompt, image]
            )
            return response.text if response.text else ""
        except Exception as e:
            print(f"Error analyzing page: {e}")
            return ""

    @sleep_and_retry
    @limits(calls=60, period=60)
    def create_embeddings(self, data: str):
        """Create embeddings with rate limiting - exactly as in Google's example"""
        time.sleep(1)
        return self.client.models.embed_content(
            model=Config.TEXT_EMBEDDING_MODEL_ID,
            contents=data,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

    def find_best_passage(self, query: str, dataframe: pd.DataFrame) -> dict:
        """Find the most relevant passage for a query"""
        try:
            query_embedding = self.client.models.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_ID,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            
            dot_products = np.dot(np.stack(dataframe['Embeddings']), 
                                query_embedding.embeddings[0].values)
            idx = np.argmax(dot_products)
            content = dataframe.iloc[idx]['Original Content']
            return {
                'page': content['page_number'],
                'content': content['content']
            }
        except Exception as e:
            print(f"Error finding best passage: {e}")
            return {'page': 0, 'content': ''}

    def make_answer_prompt(self, query: str, passage: dict) -> str:
        """Create prompt for answering questions"""
        escaped = passage['content'].replace("'", "").replace('"', "").replace("\n", " ")
        return textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
                                 You are answering questions about a research paper. 
                                 Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
                                 However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
                                 strike a friendly and conversational tone. 
                                 If the passage is irrelevant to the answer, you may ignore it.
                                 
                                 QUESTION: '{query}'
                                 PASSAGE: '{passage}'
                                 
                                 ANSWER:
                              """).format(query=query, passage=escaped)

class RAGApplication:
    """Main RAG application class"""
    
    def __init__(self, api_key: str):
        self.gemini_client = GeminiClient(api_key)
        self.data_df = None
        
    def process_pdf(self, pdf_path: str):
        """Process PDF using Gemini's vision capabilities"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Convert PDF pages to images
        images = PDFProcessor.pdf_to_images(pdf_path)
        
        # Analyze each page
        page_contents = []
        page_analyses = []
        
        print("Analyzing PDF pages...")
        for i, image in enumerate(tqdm(images)):
            content = self.gemini_client.analyze_page(image)
            if content:
                # Store both the analysis and the content
                page_contents.append({
                    'page_number': i+1,
                    'content': content
                })
                page_analyses.append(content)
        
        if not page_analyses:
            raise ValueError("No content could be extracted from the PDF")
            
        # Create dataframe
        self.data_df = pd.DataFrame({
            'Original Content': page_contents,
            'Analysis': page_analyses
        })
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = []
        try:
            for text in tqdm(self.data_df['Analysis']):
                embeddings.append(self.gemini_client.create_embeddings(text))
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            time.sleep(10)
            
        _embeddings = []
        for embedding in embeddings:
            _embeddings.append(embedding.embeddings[0].values)
            
        self.data_df['Embeddings'] = _embeddings
        
    def answer_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        """Answer a list of questions using the processed data"""
        if self.data_df is None:
            raise ValueError("Please process a PDF first using process_pdf()")
            
        answers = []
        for question in questions:
            try:
                passage = self.gemini_client.find_best_passage(question, self.data_df)
                prompt = self.gemini_client.make_answer_prompt(question, passage)
                response = self.gemini_client.client.models.generate_content(
                    model=Config.MODEL_NAME,
                    contents=prompt
                )
                answers.append({
                    'question': question,
                    'answer': response.text,
                    'source': f"Page {passage['page']}\nContent: {passage['content']}"
                })
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                answers.append({
                    'question': question,
                    'answer': f"Error generating answer: {str(e)}",
                    'source': "Error"
                })
            
        return answers

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variable and add debugging
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # Debug information
    print("Environment variables available:", [key for key in os.environ.keys() if 'API' in key])
    print("API key length:", len(api_key) if api_key else "No API key found")
    
    if not api_key:
        # Try alternative environment variable names
        alternative_names = ['GEMINI_API_KEY', 'GOOGLE_GEMINI_KEY', 'GEMINI_KEY']
        for name in alternative_names:
            api_key = os.getenv(name)
            if api_key:
                print(f"Found API key in {name}")
                break
    
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable in your environment or .env file")
    
    try:
        # Test the API key first
        print("Testing API key...")
        test_client = genai.Client(api_key=api_key)
        test_response = test_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Hello, this is a test message. Please respond with 'API key is working' if you receive this."
        )
        print("API test response:", test_response.text)
        
        # Initialize application
        app = RAGApplication(api_key)
        
        # Process PDF
        pdf_path = "med_gemini.pdf"  # Replace with your PDF path
        print(f"\nProcessing PDF: {pdf_path}")
        app.process_pdf(pdf_path)
        
        # Example questions
        questions = [
            "Summerize this document?"
        ]
        
        # Get answers
        print("\nAnswering questions...")
        answers = app.answer_questions(questions)
        
        # Print results
        for result in answers:
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Source: {result['source']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()