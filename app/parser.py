# from llama_parse import LlamaParse
# from PyPDF2 import PdfReader 
# import os

# def parse_pdf(pdf_path: str) -> str:
#      """Try parsing PDF using LlamaParse, fallback to PyPDF2."""
#      llama_key = os.getenv("LLAMA_CLOUD_API_KEY") #gets the llama api key from the environment variables

#      if llama_key:
#         parser = LlamaParse(api_key=llama_key) #instantiates the llama parse object
#         docs = parser.load_data(pdf_path) #loads the data from the pdf file
#         return "\n\n".join([d.text for d in docs]) #returns the text from the pdf file 

#      else:
#         reader = PdfReader(pdf_path) #instantiates the pdf reader object
#         text = "" #initializes the text variable
#         for page in reader.pages: #iterates through the pages of the pdf file
#             text += page.extract_text() or "" #extracts the text from the pdf file
#         return text #returns the text from the pdf file


from PIL import Image
import pytesseract
from pdf2image import convert_from_path




def parse_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    from PyPDF2 import PdfReader
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_pdf_with_images(pdf_path: str) -> dict:
    """Parse PDF and extract both text and images."""
    
    # Extract text (existing functionality)
    text_content = parse_pdf(pdf_path)
    
    # Extract images from PDF
    images = extract_images_from_pdf(pdf_path)
    
    # OCR images to text
    image_texts = []
    for i, img in enumerate(images):
        try:
            # Convert PIL image to text using OCR
            img_text = pytesseract.image_to_string(img)
            if img_text.strip():
                image_texts.append(f"Image {i+1} content: {img_text.strip()}")
        except Exception as e:
            print(f"OCR failed for image {i+1}: {e}")
    
    return {
        "text_content": text_content,
        "image_content": image_texts,
        "total_images": len(images)
    }

def extract_images_from_pdf(pdf_path: str) -> list:
    """Extract images from PDF pages."""
    try:
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path)
        images = []
        
        for page in pages:
            # Extract images from each page
            # This is a simplified approach - you might want more sophisticated image extraction
            images.append(page)
        
        return images
    except Exception as e:
        print(f"Image extraction failed: {e}")
        return []