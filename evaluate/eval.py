from groq import Groq,RateLimitError
from dotenv import load_dotenv
from evaltools import oneshotex,userprom,sys_prompt,unwanted,groq_dict
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from mr_teacher import is_correct 
import pandas as pd  
import os
import re
import json
import time
import argparse
import sys

GENERATOR_DIR = Path(__file__).resolve().parent
QA_PAIRS_DIR = GENERATOR_DIR.parent / "qa_pairs"
EVAL_OUTPUT_DIR =GENERATOR_DIR.parent / "eval_output"
load_dotenv(GENERATOR_DIR.parent/".env")
API_KEY= os.getenv("GROQ_API_KEY")
clients = {
    "groq": OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"),
    "gemini": OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
}

qas = {
    f"{i+1}":f"{QA_PAIRS_DIR}/{fil}"
 for i,fil in enumerate(f for f in os.listdir(QA_PAIRS_DIR) if f.endswith(".json"))}

diffs={
    "easy":1,"medium":2,"hard":3
}
parser = argparse.ArgumentParser()
def main(file):
    start = time.perf_counter()
    Sample = Process(rows = 15 , file = file).grab()
    with open(file) as f:
        chunk = f.read(10000)
    difficulty = re.search(r'"difficulty"\s*:\s*"([^"]+)"', chunk)
    parser_ = re.search(r'"parser"\s*:\s*"([^"]+)"', chunk)
    chunker = re.search(r'"chunker"\s*:\s*"([^"]+)"', chunk)

    difficulty = difficulty.group(1) if difficulty else None
    parser_ = parser_.group(1) if parser_ else None
    chunker = chunker.group(1) if chunker else None

    Gret = Chatt(pairs = Sample, model = groq_dict.get("1"),provider = "groq").test_it()
    df =pd.DataFrame(Gret)
    if diffs.get(difficulty) == 1: 
        df["correctness_score"] = df.apply(lambda r: Tools.correctness_score(r["answer"], r["model_answer"]), axis=1)
    else:
        df =is_correct(df)
    df["correct"] = df["correctness_score"] >= 0.90  
    df[[ "file_analyzed","difficulty","question", "answer", "model_answer",
        "model_certainty","source","correct","correctness_score","ground_score",
        "parser","chunker"]].to_json(EVAL_OUTPUT_DIR/f"{difficulty}_{parser_}_{chunker}_{len(df)}.json",
         orient='records', index=False)
    elapsed = time.perf_counter() -start
    print(f"Completed Action! -- Total runtime: {elapsed:.2f} seconds")

class Tools:
        @staticmethod
        def extract_json(text):
            if not text or not isinstance(text, str):
                return {}
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            try:
                result = json.loads(text)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
            matches = re.findall(r"\{.*?\}", text, re.DOTALL)
            for candidate in reversed(matches):
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue
            return {}
        @staticmethod
        def normalize_number(s):
            if s is None:
                return None
            s = str(s).strip().lower()
            
            for item in  unwanted:
                s = s.replace(item, "")
            try:
                return float(s)
            except ValueError:
                return None
        @staticmethod
        def correctness_score(gold, pred):
            g, p = Tools.normalize_number(gold), Tools.normalize_number(pred)
            if g is None or p is None:
                return 1.0 if str(gold).strip().lower() == str(pred).strip().lower() else 0.0
            if g == 0:
                return 1.0 if abs(p) < 0.5 else 0.0
            rel_err = abs(p - g) / abs(g)
            x = max(0.0, 1.0 - rel_err) 
            return round(x,2)
        def safe_certainty(parsed):
            val = parsed.get("certainty")
            if val is None or str(val).strip().lower() in ("", "none", "null", "n/a"):
                return ""
            return str(val).strip()
class Chatt:
    def __init__(self,pairs,model,provider):
        self.pairs= pairs
        self.model = model
        self.provider= provider

    def query(self,model,context, question):
        prompt = userprom.format(
            one_shot_example=oneshotex,
            context=context,
            question=question,
            max_tokens=4096, 
        )
        for attempt in range(3):
            try:
                response = clients[self.provider].chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except RateLimitError:
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"failed to call {model}: {e}")
                return None
    def test_it(self):
        results = []
        for item in tqdm(self.pairs, desc=f"Querying {self.model}", unit="row"):
            raw_response = self.query(context=item["context"], question=item["question"], model=self.model)
            parsed = Tools.extract_json(raw_response)
            results.append({
                **item,
                "raw_response": raw_response,
                "model_answer": parsed.get("answer", ""),
                "model_steps": parsed.get("steps", ""),
                "model_certainty": Tools.safe_certainty(parsed),
                "source": parsed.get("source", "")
            })
        return results
class Process:
    def __init__(self,rows,file,):
        self.data = file
        self.rows = rows
    def grab(self):
        df = pd.read_json(self.data,lines=True)
        df_slice = df[:self.rows]
        QAs= []
        for i in range(len(df_slice)):
            QAs.append({
                "question": f'{df_slice.loc[i,"question"]}',
                "answer": f'{df_slice.loc[i,"answer"]}',
                "context": f'{df_slice.loc[i,"context"]}',
                "id" :f'{df_slice.loc[i,"page_num"]}',
                "file_analyzed":f'{df_slice.loc[i,"file_analyzed"]}',
                "difficulty":f'{df_slice.loc[i,"difficulty"]}',
                "ground_score":f'{df_slice.loc[i,"ground_score"]}',
                "parser":f'{df_slice.loc[i,"parser"]}',
                "chunker":f'{df_slice.loc[i,"chunker"]}',
            })
        return QAs
    
if __name__ == "__main__":
    try:
        for i, fil in enumerate(f for f in os.listdir(QA_PAIRS_DIR) if f.endswith(".json")):
            (print(f"{i+1}. {fil}"))
        while True:
            choice = input("Select the file number: ").strip()
            if choice in qas:
                file = qas[choice]
                break 
            print("Invalid selection.") 
        main(file=file)
    except KeyboardInterrupt:
        sys.exit(0)
    