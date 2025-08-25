from llama_parse import LlamaParse
from PyPDF2 import PdfReader 
import os

def parse_pdf(pdf_path: str) -> str:
     """Try parsing PDF using LlamaParse, fallback to PyPDF2."""
     llama_key = os.getenv("LLAMA_CLOUD_API_KEY") #gets the llama api key from the environment variables

     if llama_key:
        parser = LlamaParse(api_key=llama_key) #instantiates the llama parse object
        docs = parser.load_data(pdf_path) #loads the data from the pdf file
        return "\n\n".join([d.text for d in docs]) #returns the text from the pdf file 

     else:
        reader = PdfReader(pdf_path) #instantiates the pdf reader object
        text = "" #initializes the text variable
        for page in reader.pages: #iterates through the pages of the pdf file
            text += page.extract_text() or "" #extracts the text from the pdf file
        return text #returns the text from the pdf file





