# Turkish Question Answering System for Gebze Technical University Student Rules

This project implements a question-answering (QA) system that can answer questions based on the official student rules and regulations of Gebze Technical University (GTU). The system is designed to understand Turkish natural language queries and provide accurate answers extracted directly from the university's official documents.

## Project Structure

*   **`qa_system.py`**: Main application script for the QA system, including the Streamlit interface.
*   **`pdf_to_vector_store.py`**: Script for creating a vector store from PDF documents.
*   **`vector_store.py`**: Class for interacting with the vector store.
*   **`train_model.py`**: Script for fine-tuning the BERT model on the QA dataset.
*   **`qa_dataset_generator.py`**: Script for generating the QA dataset from PDF documents using the Gemini API.
*   **`evaluate_model.py`**: Script for evaluating the fine-tuned model.
*   **`data/`**: Directory containing the PDF documents of GTU's student rules.
*   **`qa_results/`**: Directory to store the generated QA dataset (CSV files).
*   **`vector_store/`**: Directory to store the Chroma vector database.
*   **`gtu_turkish_qa_model/`**: Directory containing the fine-tuned model and tokenizer. (Can be downloaded from Hugging Face - see instructions below)

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**

    *   **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

    *   **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Download the fine-tuned model:**
    You can download the fine-tuned model from the Hugging Face Model Hub in two ways:

    **a) Using the Hugging Face Website Interface:**

    *   Go to [https://huggingface.co/ResitAydin/gtu-turkish-qa](https://huggingface.co/ResitAydin/gtu-turkish-qa).
    *   Click on the "Files" tab.
    *   Download all the files in the repository and place them inside the `gtu_turkish_qa_model` folder in your project.

    **b) Using `git clone` (Recommended, especially if you have Git LFS installed):**

    *   Make sure you have Git LFS installed. Run `git lfs install` in your terminal.
    *   Run the following command:

        ```bash
        git clone https://huggingface.co/ResitAydin/gtu-turkish-qa
        ```
        This will create a folder named `gtu-turkish-qa`.
    *   Rename this folder to `gtu_turkish_qa_model`
    *   Move the `gtu_turkish_qa_model` folder into the root directory of your cloned project repository.

## Running the Application

1. **Prepare the data (if you haven't already):**

    *   Place the PDF files of GTU's student rules in the `data/` directory. You can find them here: [https://www.gtu.edu.tr/icerik/1479/592/lisans-yonetmelik-ve-yonergeler.aspx](https://www.gtu.edu.tr/icerik/1479/592/lisans-yonetmelik-ve-yonergeler.aspx)
    *   Run the `qa_dataset_generator.py` script to generate the QA dataset:

        ```bash
        python qa_dataset_generator.py
        ```

    *   Run the `pdf_to_vector_store.py` script to create the vector store:

        ```bash
        python pdf_to_vector_store.py
        ```

2. **Run the QA system:**

    ```bash
    streamlit run qa_system.py
    ```

    This will open the Streamlit application in your web browser. You can then start asking questions about GTU's student rules.

## Evaluation

To evaluate the model, you can use the `evaluate_model.py` script. You'll need a test dataset in CSV format with "question," "answer," and "context" columns. Place your test dataset in the root directory of the project. Then run:

```bash
python evaluate_model.py
