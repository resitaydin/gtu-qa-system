import os
import PyPDF2
import google.generativeai as genai
import pandas as pd
import json
from typing import List, Dict, Tuple
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document
import json
from typing import Dict

class QADatasetGenerator:
    def __init__(self):
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def split_text_into_chunks(self, text: str) -> List[Document]:
        # Simplified splitting logic
        docs = self.splitter.create_documents([text])
        return docs

    def clean_and_preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text before chunking
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove page numbers and headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip page numbers
            if line.strip().isdigit():
                continue
            # Skip lines that are likely headers/footers
            if len(line.strip()) < 5 or line.strip().isupper():
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def generate_qa_pairs(self, text: str) -> List[Dict]:
        print(f"\nDEBUG: Starting QA generation. Input text length: {len(text)}")
        prompt = f"""
        Aşağıdaki metinden 30-60 tane soru üretin.
        Uzun dökümanlarda daha çok soru üret. 
        Her soru için şu formatı kullanın:

        SORU:
        CEVAP:
        KAYNAK: (Lütfen tam bağlamı, metnin tamamını yazın, sadece madde numarası değil)

        - Yönetmelik maddelerine spesifik olmayan, genel anlayışa yönelik sorular üretin.
        - "Bu, şu" gibi placeholder ifadeler kullanmayın, soruları mümkün olduğunca açık ve anlaşılır yazın.
        - Gerekirse başlıklara bakarak soruları oluşturun.

        ASLA SORULMAYACAK SORU ÖRNEKLERİ:
        ❌ "Bu yönetmeliğin X. maddesinde ne yazıyor?"
        ❌ "Bu yönetmelik hangi tarihte yürürlüğe girmiştir?"
        ❌ "Bu yönetmelikte hangi durumda Y gerekir?"
        ❌ "Kaçıncı maddede W konusu geçiyor?"
        ❌ "Bu yönetmelik hangi yasal düzenlemeye dayanarak hazırlanmıştır?"
        
        SORULMASI GEREKEN SORU ÖRNEKLERİ:
        ✅ "Üniversiteye kayıt olmak için hangi sınava girmek gerekir?"
        ✅ "Yatay geçişler hangi yönetmeliğe göre yapılır?"

        İşlenecek metin:
        {text}
        """
        
        try:
            print("DEBUG: Sending request to Gemini")
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            print(f"\nDEBUG: Raw response from Gemini:\n{response_text[:500]}...")
            print(f"DEBUG: Response length: {len(response_text)}")
            
            qa_pairs = []
            current_qa = {}
            
            for line in response_text.split('\n'):
                line = line.strip()
                # Handle markdown format
                if line.startswith('**SORU'):
                    if current_qa:
                        qa_pairs.append(current_qa)
                    current_qa = {'question': line.split(':**')[1].strip()}
                elif line.startswith('**CEVAP'):
                    current_qa['answer'] = line.split(':**')[1].strip()
                elif line.startswith('**KAYNAK'):
                    current_qa['context'] = line.split(':**')[1].strip()
                # Handle original format
                elif line.startswith('SORU:'):
                    if current_qa:
                        qa_pairs.append(current_qa)
                    current_qa = {'question': line[6:].strip()}
                elif line.startswith('CEVAP:'):
                    current_qa['answer'] = line[6:].strip()
                elif line.startswith('KAYNAK:'):
                    current_qa['context'] = line[7:].strip()
            
            if current_qa and all(key in current_qa for key in ['question', 'answer', 'context']):
                qa_pairs.append(current_qa)

            print(f"DEBUG: Generated {len(qa_pairs)} QA pairs")
            return qa_pairs
            
        except Exception as e:
            print(f"DEBUG: Error in generate_qa_pairs: {str(e)}")
            print(f"DEBUG: Error type: {type(e)}")
            import traceback
            print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
            return []

    def test_single_pdf(self, pdf_path: str, verbose: bool = True) -> Dict:
        test_results = {
            'pdf_path': pdf_path,
            'success': False,
            'error': None,
            'stats': {},
            'qa_pairs': []
        }
        
        try:
            text = self.extract_text_from_pdf(pdf_path)
            cleaned_text = self.clean_and_preprocess_text(text)
            qa_pairs = self.generate_qa_pairs(cleaned_text)
            
            test_results['stats'] = {
                'total_qa_pairs': len(qa_pairs)
            }
            
            test_results['success'] = True
            test_results['qa_pairs'] = qa_pairs
            
        except Exception as e:
            test_results['success'] = False
            test_results['error'] = str(e)
        
        return test_results
    
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        all_qa_pairs = []
        processed_files = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                try:
                    pdf_path = os.path.join(input_dir, filename)
                    print(f"\nProcessing: {filename}")
                    
                    # Extract and process text
                    text = self.extract_text_from_pdf(pdf_path)
                    cleaned_text = self.clean_and_preprocess_text(text)
                    qa_pairs = self.generate_qa_pairs(cleaned_text)
                    
                    all_qa_pairs.extend(qa_pairs)
                    processed_files.append({
                        'filename': filename,
                        'qa_pairs_count': len(qa_pairs)
                    })
                    
                    # Save individual file results
                    df_single = pd.DataFrame(qa_pairs)
                    single_output = os.path.join(output_dir, f"{filename[:-4]}_qa.csv")
                    df_single.to_csv(single_output, index=False, encoding='utf-8')
                    
                    print(f"Generated {len(qa_pairs)} QA pairs for {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        # Save combined results
        if all_qa_pairs:
            combined_df = pd.DataFrame(all_qa_pairs)
            combined_output = os.path.join(output_dir, "all_qa_pairs.csv")
            combined_df.to_csv(combined_output, index=False, encoding='utf-8')
            
            # Save processing summary
            summary_df = pd.DataFrame(processed_files)
            summary_output = os.path.join(output_dir, "processing_summary.csv")
            summary_df.to_csv(summary_output, index=False, encoding='utf-8')
            
            print(f"\nTotal QA pairs generated: {len(all_qa_pairs)}")
            print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    # Directory paths
    input_dir = "data"  # folder containing PDFs
    output_dir = "qa_results"  # folder for output CSV files
    
    generator = QADatasetGenerator()
    generator.process_directory(input_dir, output_dir)