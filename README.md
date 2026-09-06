# QA Pair Generator & Evaluator

A small pipeline that:
1. Parses PDF filings into text/table chunks
2. Generates QA pairs from those chunks using an LLM (via DSPy)
3. Queries a model to answer the generated questions
4. Judges the answers for correctness against the ground truth

## Project Structure

```
.
├── generator/
│   ├── runner.py     # Entry point: parse PDF -> chunk -> generate QA pairs
│   ├── pdfpipe.py     # Chunking + context-building orchestration
│   └── pdfparse.py    # PDF parsing backends (PyPDF2, pymupdf, pdfplumber)
├── evaluate/
│   ├── eval.py        # Entry point: run generated QA pairs through a model + score them
│   ├── evaltools.py   # Prompts, parsing helpers, and answer-normalization utilities
│   └── mr_teacher.py  # LLM-as-judge for factual correctness scoring
├── sample_filings/    # (expected) input PDF files
├── qa_pairs/          # (expected) generated QA pairs output (JSON)
├── eval_output/       # (expected) evaluation results output (JSON)
└── requirements.txt
```

Both `generator/runner.py` and `evaluate/eval.py` locate the project root as
`Path(__file__).resolve().parent.parent` (i.e. one level above their own folder),
and expect `.env`, `sample_filings/`, `qa_pairs/`, and `eval_output/` there.

## Requirements

- Python 3.10+
- A `.env` file in the project root (one level above `generator/` and `evaluate/`) containing:
  ```
  GEMINI_API_KEY=your_gemini_key
  GROQ_API_KEY=your_groq_key
  ```
- Dependencies listed in `requirements.txt`

Install with:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate QA pairs from a PDF

```bash
cd generator
python runner.py -d easy   # or medium / hard
```

You'll be prompted to interactively choose:
- Which PDF (from `sample_filings/`)
- A PDF parsing library (PyPDF2, pymupdf, pdfplumber)
- A model (Gemini Flash, or Groq-hosted GPT-OSS 120b/20b)
- A chunking strategy (recursive, token, semantic, sentence, neural)
- Chunk size and number of questions to generate

Output is saved to `qa_pairs/<difficulty>.json`.

### 2. Evaluate generated QA pairs

```bash
cd evaluate
python eval.py
```

This pulls a sample from `qa_pairs/hard.json`, queries a model for answers,
then scores correctness either with a simple numeric comparison (`Tools.correctness_score`)
or an LLM judge (`mr_teacher.is_correct`), depending on difficulty.

Output is saved to `eval_output/test.json`.

## Notes

- `generator/runner.py` and `evaluate/eval.py` are meant to be run as standalone scripts
  (both have `if __name__ == "__main__":` entry points).
- Ground-score filtering in `generator/runner.py` (`ground_thresh=0.35`) drops generated QA pairs
  whose answer doesn't overlap enough with the source context, as a basic hallucination check.
