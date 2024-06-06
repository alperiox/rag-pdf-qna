# RAG-based pdf Q&A application

The app uses Streamlit to render the user interface.

## Installation

Clone the project:

```bash
git clone https://github.com/alperiox/rag-pdf-qna
```

Install the poetry project:

```bash
poetry install
```

Save your GCP API key to the dotenv file with `GOOGLE_API_KEY`:

```bash
# filename: .env
GOOGLE_API_KEY=<api-key>
```

Then you can run the project in the poetry shell:

```bash
streamlit run app.py
```

Future improvements:

- [ ] Let the LLM choose the relevant database from the uploaded documents
- [ ] Adding support for more LLMs, including local ones
- [ ] UI features for cleaning the history, etc.
