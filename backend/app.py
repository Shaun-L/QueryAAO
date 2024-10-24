from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
import pickle
import faiss
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
chat_model = ChatOpenAI(api_key=OPEN_AI_KEY, model="gpt-4-turbo")

# Load the FAISS index and metadata
index_file_path = "./uscis_data/legal_embeddings.index"
metadata_file_path = "./uscis_data/metadata.pkl"
index = faiss.read_index(index_file_path)

with open(metadata_file_path, 'rb') as f:
    metadata = pickle.load(f)

# Load the LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def create_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

@app.route('/setup', methods=['POST'])
def setup():
    try:
        setup_application()  # Call your setup_application function
        return jsonify({'message': 'Application setup successfully!'}), 200
    except Exception as e:
        print(f"Error during setup: {e}")
        return jsonify({'error': 'An error occurred while setting up the application.'}), 500


@app.route('/query', methods=['POST'])
def query():
    if not os.path.exists(index_file_path) or not os.path.exists(metadata_file_path):
        return jsonify({'error': 'The index and metadata files are missing. Please run the setup script first.'}), 400
    
    data = request.json
    query_text = data.get('query')
    
    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400

    try:
        # Generate embedding for query
        query_embedding = create_embeddings(query_text, tokenizer, model)
        query_embedding = np.array([query_embedding], dtype='float32')
        D, I = index.search(query_embedding, 5)

        file_contents = ""
        for idx in I[0]:
            if idx in metadata:
                file_path = metadata[idx]['file_path']
                try:
                    file_contents += f"\n\n-----\n\n{extract_text_from_pdf(file_path)}"
                except UnicodeDecodeError as e:
                    print(f"Error reading file {file_path}: {e}")

        # Create prompt and run model
        PROMPT_TEMPLATE = """
            Answer the question based only on the following context:

            {context}

            -----

            Answer the question based on the above context: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=file_contents, question=query_text)

        response_text = chat_model.predict(prompt)

        return jsonify({'response': response_text})

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'An error occurred while processing the query.'}), 500

# Function to download PDF
def _download_pdf(url, folder_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdf_name = url.split('/')[-1]
            pdf_path = os.path.join(folder_path, pdf_name)
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded: {pdf_name}")
        else:
            print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Function to get PDF links from a page
def _get_pdf_links(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_links = []
    for link in soup.find_all('a', href=True):
        if link['href'].endswith('.pdf'):
            pdf_links.append(link['href'])
    return pdf_links


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def create_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def _get_uscis_pdfs():
    ##GETTING PDFS

    # Base URL and folder to save PDFs
    base_url = "https://www.uscis.gov"
    folder_path = "uscis_pdfs_all"
    page_url_template = f"{base_url}/administrative-appeals/aao-decisions/aao-non-precedent-decisions?uri_1=19&m=All&y=All&items_per_page=100&page={{}}"

    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # set of 43 for all pages, total of 4.3 GB of pdf data
    for i in range(43):
        print(f"We are on page number #{i}")
        page_url = page_url_template.format(i)
        pdf_links = _get_pdf_links(page_url)
        for pdf_link in pdf_links:
            # Construct full URL for each PDF
            full_pdf_url = base_url + pdf_link
            _download_pdf(full_pdf_url, folder_path)
    
def _vectorize_pdfs():
    pdf_folder = "uscis_pdfs_all"
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    if os.path.exists(metadata_file_path):
        os.remove(metadata_file_path)

    if os.path.exists(index_file_path):
        os.remove(index_file_path)

    dimension = 768  # LegalBERT embedding dimension
    index = faiss.IndexFlatL2(dimension)
    
    metadata = {}

    metadata_counter = len(metadata)

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        
        text = extract_text_from_pdf(pdf_file)
        if text:
            embedding = create_embeddings(text, tokenizer, model)
            
            # Add the embedding to the FAISS index
            index.add(np.array([embedding], dtype='float32'))
            
            # Update metadata
            metadata[metadata_counter] = {
                'file_path': pdf_file,
                'document_id': metadata_counter
            }
            
            # Save the updated metadata
            with open(metadata_file_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"Successfully stored embedding for {pdf_file} at index {metadata_counter}")
            metadata_counter += 1
    faiss.write_index(index, index_file_path)
    print(f"FAISS index saved at: {index_file_path}")
    print(f"Metadata saved at: {metadata_file_path}")

def setup_application():
    # Make sure required directories exist
    if not os.path.exists("uscis_data"):
        os.makedirs("uscis_data")

    # Call the function to download PDFs
    _get_uscis_pdfs()
    
    # Call the function to vectorize PDFs
    _vectorize_pdfs()
    
    print("Setup complete, You can now run the app.")


if __name__ == '__main__':
    app.run(debug=True)