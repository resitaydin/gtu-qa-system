from vector_store import VectorStore
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datetime import datetime
import numpy as np
import os
import time

class TurkishQASystem:
    def __init__(self, model_path='./gtu_turkish_qa', vector_store_path='vector_store'):
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {os.path.abspath(model_path)}")
            
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

    def get_answer(self, question, context, max_length=512):
        """Extract answer from context using the QA model with improved answer validation"""
        # Prepare the inputs
        encoding = self.tokenizer(
            question,
            context,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Get the most likely beginning and end of the answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Get top-k start and end positions
        k = 3
        top_start = torch.topk(start_scores[0], k)
        top_end = torch.topk(end_scores[0], k)
        
        # Try different start-end combinations
        best_answer = ""
        best_score = float('-inf')
        
        for start_idx, start_score in zip(top_start.indices, top_start.values):
            for end_idx, end_score in zip(top_end.indices, top_end.values):
                if start_idx <= end_idx and end_idx - start_idx < 50:  # Reasonable answer length
                    score = start_score + end_score
                    if score > best_score:
                        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][start_idx:end_idx+1])
                        answer = self.tokenizer.convert_tokens_to_string(tokens)
                        
                        # Basic answer validation
                        if len(answer.split()) >= 2 and not any(x in answer for x in ['[CLS]', '[SEP]', '[PAD]']):
                            best_answer = answer
                            best_score = score
        
        return best_answer if best_answer else "Üzgünüm, bağlam içerisinde uygun bir cevap bulamadım."
    
    def process_query(self, question, verbose=False):
        """Enhanced query processing with improved context handling"""
        start_time = time.time()
        
        # Get relevant context with more results
        results = self.vector_store.search(question, k=8)
        if not results:
            return {
                'answer': "Üzgünüm, bu soruyla ilgili bir bilgi bulamadım.",
                'confidence': 0,
                'context': None,
                'processing_time': time.time() - start_time
            }
        
        # Improved context combination
        contexts = []
        weights = []
        total_length = 0
        max_context_length = 1000  # Maximum context length
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        for doc, score in results:
            # Normalize score to 0-1 range
            weight = (score + 1) / 2
            
            # Only include contexts with good relevance
            if weight > 0.3:  # Increased threshold
                context_text = doc.page_content.strip()
                context_length = len(context_text.split())
                
                if total_length + context_length <= max_context_length:
                    contexts.append(context_text)
                    weights.append(weight)
                    total_length += context_length
        
        if not contexts:
            return {
                'answer': "Üzgünüm, bu soruyla ilgili yeterince güvenilir bilgi bulamadım.",
                'confidence': 0,
                'context': None,
                'processing_time': time.time() - start_time
            }
        
        # Combine contexts with weights
        weights = np.array(weights) / sum(weights)
        combined_context = " ".join(
            ctx for ctx, w in zip(contexts, weights)
            if w > 0.1  # Ensure minimal contribution
        )
        
        # Get answer with validation
        answer = self.get_answer(question, combined_context)
        
        # Calculate confidence based on multiple factors
        answer_confidence = 0.8 if len(answer.split()) >= 3 else 0.5  # Longer answers more likely to be valid
        context_confidence = max(weights)
        confidence = answer_confidence * context_confidence
        
        result = {
            'answer': answer,
            'confidence': confidence,
            'processing_time': time.time() - start_time,
            'context': combined_context if verbose else None
        }
        
        return result

# Streamlit app
import streamlit as st
import time

def create_streamlit_app():
    st.set_page_config(page_title="GTÜ Soru-Cevap Sistemi", layout="wide")
    
    st.title("GTÜ Soru-Cevap Sistemi")
    st.markdown("---")
    
    # Initialize QA system
    @st.cache_resource
    def load_qa_system():
        return TurkishQASystem()
    
    try:
        qa_system = load_qa_system()
        st.success("Sistem başarıyla yüklendi!")
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {str(e)}")
        st.stop()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("Detaylar"):
                    st.write(f"Güven: {message['metadata']['confidence']:.1f}%")
                    st.write(f"İşlem süresi: {message['metadata']['processing_time']:.2f} saniye")
    
    # Chat input
    if prompt := st.chat_input("Sorunuzu yazın..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor..."):
                response = qa_system.process_query(prompt, verbose=True)
                
                st.markdown(response['answer'])
                
                # Add response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "metadata": {
                        "confidence": response['confidence'] * 100,
                        "processing_time": response['processing_time']
                    }
                })
                
                # Show metadata in expander
                with st.expander("Detaylar"):
                    st.write(f"Güven: {response['confidence']*100:.1f}%")
                    st.write(f"İşlem süresi: {response['processing_time']:.2f} saniye")
                    if response['context']:
                        st.write("Kullanılan Bağlam:")
                        st.text(response['context'])

if __name__ == "__main__":
    create_streamlit_app()