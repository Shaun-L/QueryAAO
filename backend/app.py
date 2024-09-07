from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
import pickle
import faiss
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
OPEN_AI_KEY = "" #insert api key here
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

@app.route('/query', methods=['POST'])
def query():
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


if __name__ == '__main__':
    app.run(debug=True)