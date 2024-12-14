# PDFGenius with Gemini 2.0 Flash

Turn any PDF into an interactive knowledge base using Google's latest `gemini-2.0-flash-exp` model. This RAG (Retrieval-Augmented Generation) application leverages advanced vision-language capabilities to analyze PDFs and answer questions about their content.

## Features

- 🚀 Uses the latest `gemini-2.0-flash-exp` model
- 📄 Processes PDFs using vision-language understanding
- 🔍 RAG implementation for information retrieval
- 💡 Context-aware responses

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/lesteroliver911/pdfgenius-gemini-2.0-flash-exp.git
cd pdfgenius-gemini-2.0-flash-exp
```

2. Install dependencies:
```bash
pip install google-generativeai pillow numpy pandas tqdm PyMuPDF ratelimit python-dotenv
```

3. Set up your environment:
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. Run the application:
```bash
python app.py
```

## Example Usage

```python
questions = [
    "Summarize this document?",
    "What are the key findings?",
    # Add your questions here
]
```

## Requirements

- Python 3.8+
- Google Gemini API key
- PDF document for analysis

## License

MIT

## Acknowledgments

Built using Google's Gemini 2.0 Flash model and inspired by the Multimodal RAG implementation from Google's generative AI examples.
