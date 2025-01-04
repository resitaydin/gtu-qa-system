import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import pandas as pd
from collections import Counter
import string
import re

class ModelEvaluator:
    def __init__(self, model_path='gtu_turkish_qa'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def normalize_text(self, text):
        """Remove punctuation and convert to lowercase"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
        return ' '.join(text.split())

    def compute_exact_match(self, prediction, ground_truth):
        """Compute exact match score"""
        return int(self.normalize_text(prediction) == self.normalize_text(ground_truth))

    def compute_f1_score(self, prediction, ground_truth):
        """Compute F1 score based on word overlap"""
        prediction_tokens = self.normalize_text(prediction).split()
        ground_truth_tokens = self.normalize_text(ground_truth).split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def get_answer(self, question, context):
        """Extract answer from context using the model"""
        # Ensure inputs are strings
        question = str(question) if question is not None else ""
        context = str(context) if context is not None else ""
        
        # Add error checking
        if not question.strip() or not context.strip():
            return ""
            
        try:
            inputs = self.tokenizer(
                question,
                context,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                
                start_idx = torch.argmax(start_scores)
                end_idx = torch.argmax(end_scores)
                
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                answer = self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
                
            return answer
        except Exception as e:
            print(f"Error processing question: '{question}' with context: '{context[:100]}...'")
            print(f"Error: {str(e)}")
            return ""

    def evaluate(self, test_data):
        """Evaluate model on test dataset"""
        exact_matches = []
        f1_scores = []
        
        for _, row in test_data.iterrows():
            predicted_answer = self.get_answer(row['question'], row['context'])
            ground_truth = row['answer']
            
            em_score = self.compute_exact_match(predicted_answer, ground_truth)
            f1_score = self.compute_f1_score(predicted_answer, ground_truth)
            
            exact_matches.append(em_score)
            f1_scores.append(f1_score)
        
        results = {
            'exact_match': sum(exact_matches) / len(exact_matches) * 100,
            'f1_score': sum(f1_scores) / len(f1_scores) * 100
        }
        
        return results

    def generate_report(self, test_data):
        """Generate detailed evaluation report"""
        print("Starting evaluation...")
        results = self.evaluate(test_data)
        
        print("\nEvaluation Results:")
        print(f"Total samples evaluated: {len(test_data)}")
        print(f"Exact Match Score: {results['exact_match']:.2f}%")
        print(f"F1 Score: {results['f1_score']:.2f}%")
        
        return results

if __name__ == "__main__":
    # Load test dataset
    test_data = pd.read_csv('test_data.csv')
    evaluator = ModelEvaluator()
    results = evaluator.generate_report(test_data)