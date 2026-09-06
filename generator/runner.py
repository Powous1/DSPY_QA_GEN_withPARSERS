from openai import OpenAI
from dotenv import load_dotenv
from pdfpipe import create_context
from pathlib import Path
import dspy
import time
import pandas as pd
import threading
import argparse
import re
import os

GENERATOR_DIR = Path(__file__).resolve().parent
load_dotenv(GENERATOR_DIR.parent/".env")
QA_PAIRS_DIR = GENERATOR_DIR.parent / "qa_pairs"
FILINGS_=GENERATOR_DIR.parent / "sample_filings"
models ={
    "gf": dspy.LM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),max_tokens=2048,num_retries=5),
    "cgl": dspy.LM(
            model="openai/openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            api_base= "https://api.groq.com/openai/v1",max_tokens=2048,num_retries=5),
            
    "cgs": dspy.LM(
            model="openai/openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY"),
            api_base= "https://api.groq.com/openai/v1",max_tokens=2048,num_retries=5),
    ### add your models here
    }

parsers ={
    "1":"pypdf2","2":"pymupdf","3":"pdfplumber"
}
model={
    "1":"gf","2":"cgl","3":"cgs"
}
chunker={
    "1":"recursive","2":"token","3": "semantic", 
    "4": "sentence","5": "neural" 
}
files = {
    f"{i+1}":f"{FILINGS_}/{fil}"
 for i,fil in enumerate(os.listdir(FILINGS_))}

parser = argparse.ArgumentParser()
def main(mod,pdf,chunk,chunk_size,file,rows):
    parser.add_argument("-d","--difficulty",
                        type = str,
                        help="'-f file.json 'or '--file file.json' to process file.json ")
    args = parser.parse_args()
    start = time.perf_counter()
    df = create_context(parser=pdf, pdf_file=file, 
                        rows=rows, chunk=chunk, chunk_size=chunk_size).get_tables()
    elapsed = time.perf_counter() - start
    print(f"Parsing completed in {elapsed:.2f}s")
    df = pd.DataFrame(df)
    start = time.perf_counter()
    qa_pairs = qa_gen(data =df,model = mod, difficulty = args.difficulty).runit()
    elapsed = time.perf_counter() - start
    print(f"QA Pair Generation completed in {elapsed:.2f}s")
    qa_pairs.to_json(f"{QA_PAIRS_DIR}/{args.difficulty}.json", orient='records', lines=True, index=False)
    return "completed"


class QA(dspy.Signature):
    difficulty: str = dspy.InputField(desc="The difficulty level of the task. Use \'easy\' for a single numeric answer, \'medium\' for a short sentence combined with a number, and \'hard\' for a longer sentence together with a numeric answer.")
    context: str = dspy.InputField(desc='The `context` field provides the source text (as a plain string) that the QA pair generator will read and use to create question‑answer pairs. It should contain the passage, article, or document from which relevant information is extracted.')
    question: str = dspy.OutputField(desc='A string containing the automatically generated question (Always limit to 1 question). This field is populated by DSPy and represents the query that will be paired with its corresponding answer in the QA pair. It should be a plain `str` value.')
    answer: str= dspy.OutputField(desc='a output field that holds the generated answer for the given question in a QA‑pair generation task.')


def ground_score(answer:str,context:str):
    def tokens(s):
        return set(re.findall(r"\w+", s.lower()))
    ans_tok = tokens(answer)
    ctx_tok = tokens(context)
    if not ans_tok:
        return 0.0
    return len(ans_tok & ctx_tok) / len(ans_tok)

class qa_gen:
    def __init__(self,data,model,difficulty):
        self.model = f'{model}'
        self.diff = f"{difficulty}"
        self.data= data        

    def runit(self,ground_thresh=0.35, num_threads: int = 2):
        
        dspy.configure(lm=models[self.model], temperature=0.7)
        loaded_program = dspy.Predict(QA)
        loaded_program.load("optimized_quiz_generator.json")
        df = self.data
        rows = list(df.itertuples())
        exec_pairs = [
            (loaded_program, dict(context=row.table, difficulty=self.diff))
            for row in rows
        ]
        parallelizer = dspy.Parallel(
            num_threads=num_threads,max_errors=len(rows),provide_traceback=True,
        )
        preds = parallelizer(exec_pairs)  
        qas = [
            {
                "id": row.doc_id,
                "type": row.type,
                "context": row.table,
                "question": pred.question,
                "answer": pred.answer,
                "page_num": row.page_num,
                "ground_score":score
            }
            for row, pred in zip(rows, preds)
            for score in [ground_score(answer=pred.answer, context=row.table) if pred is not None else None]
            if pred is not None and score is not None and score >= ground_thresh
        ]
        
        return pd.DataFrame(qas)

def live_timer(stop_event):
    """Refreshes a live stopwatch in the CLI until stopped."""
    start_time = time.perf_counter()
    while not stop_event.is_set():
        elapsed = time.perf_counter() - start_time
        print(f"\rElapsed time: [{elapsed:.1f}]-----------", end="", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    stop_signal = threading.Event()

    print("Choose which file to analyze")
    for i, fil in enumerate(os.listdir(FILINGS_)):
        (print(f"{i+1}. {fil}"))
    while True:
        choice = input("Select the file number: ").strip()
        if choice in files:
            file = files[choice]
            break 
        print("Invalid selection.") 

    print("--- Choose a PDF Library ---")
    print("1. PyPDF2, 2. pymupdf, 3. pdfplumber")
    while True:
        choice = input("Select a library number: ").strip()
        if choice in parsers:
            pdf = parsers[choice]
            break 
        print("Invalid selection.")
    print("--- Choose a Model ---")
    print("1. gemini-2.5-flash, 2. chatgpt-large, 3. chagpt-small")
    while True:
        choice = input("Select a model: ").strip()
        if choice in model:
            mod = model[choice]
            break 
        print("Invalid selection.")
    print("--- Choose a Chunking Strategy ---")
    print("1. recursive, 2. token, 3. semantic")
    print("4. sentence, 5. neural")
    while True:
        choice = input("Select a chunking strategy: ").strip()
        if choice in chunker:
            chunk = chunker[choice]
            break 
        print("Invalid selection.")  
    while True:
        try:
            user_input = input("Enter a chunking size(the default is 2000):  ").strip()
            if user_input == "":
                value = 2000
                break
            value = int(user_input)  
            break  
        except ValueError:
            print("Invalid input.")
    while True:
            try:
                user_input = input("How many questions to be proposed(the default is 10):  ").strip()
                if user_input == "":
                    rows = 10
                    break
                rows = int(user_input)  
                break  
            except ValueError:
                print("Invalid input.")           
    timer_thread = threading.Thread(target=live_timer, args=(stop_signal,), daemon=True)
    timer_thread.start()

    try:
        main(mod=mod,pdf=pdf,chunk=chunk,chunk_size = value,file=file,rows=rows)
    finally:
        stop_signal.set()
        timer_thread.join()