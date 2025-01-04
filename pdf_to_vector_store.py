from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import torch
import numpy as np
import os
import time
from datetime import datetime
from typing import List, Optional
import logging
from pathlib import Path
import gc

class TurkishBERTEmbeddings:
    def __init__(self, 
                 model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
                 batch_size: int = 32,
                 max_length: int = 512):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading {model_name} model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts and return embeddings."""
        inputs = self.tokenizer(texts, 
                              return_tensors="pt",
                              truncation=True,
                              max_length=self.max_length,
                              padding=True)
        
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use attention mask for proper averaging of padded sequences
            mask = inputs['attention_mask'].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            masked_embeddings = outputs.last_hidden_state * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        embeddings = self._batch_encode([text])
        return embeddings[0].cpu().numpy()

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of documents using batching."""
        self.logger.info("Embedding documents...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self._batch_encode(batch_texts)
            all_embeddings.extend(embeddings.cpu().numpy())
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and (i + 1) % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()
                
        return all_embeddings

class VectorStoreCreator:
    def __init__(self, 
                 data_dir: str = "data",
                 persist_directory: str = "vector_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.data_dir = Path(data_dir)
        self.persist_directory = Path(persist_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def create_vector_store(self) -> Chroma:
        start_time = time.time()
        self.logger.info("Starting vector store creation...")

        try:
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Load PDFs
            self.logger.info(f"Loading PDFs from {self.data_dir}...")
            loader = DirectoryLoader(
                str(self.data_dir),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} documents")
            
            # Split text
            self.logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = []
            for doc in tqdm(documents, desc="Splitting documents"):
                splits.extend(text_splitter.split_documents([doc]))
                
            self.logger.info(f"Created {len(splits)} text chunks")

            # Initialize embeddings
            embeddings = TurkishBERTEmbeddings(batch_size=32)

            # Create vector store
            self.logger.info("Creating vector store...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # Persist
            self.logger.info("Persisting vector store...")
            vectorstore.persist()
            
            duration = time.time() - start_time
            self.logger.info(f"Vector store creation completed in {duration:.2f} seconds")
            
            return vectorstore

        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    creator = VectorStoreCreator()
    try:
        vectorstore = creator.create_vector_store()
        print("\nProcess completed successfully! ✨")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")