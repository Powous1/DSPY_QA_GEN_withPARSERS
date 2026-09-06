import dspy
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os


GENERATOR_DIR = Path(__file__).resolve().parent
load_dotenv(GENERATOR_DIR.parent/".env")

lm= dspy.LM(model="openai/openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    api_base= "https://api.groq.com/openai/v1",max_tokens=2048,num_retries=5)

class LLMJudge(dspy.Signature):
    """Judge whether a generated answer is factually correct relative to the ground-truth answer.
    Focus on factual accuracy, not wording, formatting, or phrasing differences."""

    question: str = dspy.InputField(desc="The original question being answered.")
    actual_answer: str = dspy.InputField(desc="The ground-truth answer from the dataset.")
    generated_answer: str = dspy.InputField(desc="The model-generated answer to evaluate.")

    correctness_score: float = dspy.OutputField(
        desc="A float between 0.0 and 1.0 indicating factual correctness. "
             "1.0: the generated answer conveys the same fact/value as the actual answer "
             "(ignore formatting differences like '$1,458' vs '1458' vs '1458.00'). "
             "0.5-0.9: partially correct, close but not exact (e.g. right order of magnitude, "
             "missing a qualifier, or a minor numeric discrepancy). "
             "0.0-0.4: factually wrong, contradicts the actual answer, or is unrelated/empty. "
             "Always return a numeric value, never null or a string — use a low score to express "
             "low confidence rather than omitting the field."
    )
dspy.configure(lm=lm,temperature=0.7)
judge = dspy.ChainOfThought(LLMJudge)

def is_correct(df):
    exec_pairs = [
        (judge, dict(
            question=row.question,
            actual_answer=row.answer,
            generated_answer=row.model_answer,
        ))
        for row in df.itertuples()
    ]

    parallelizer = dspy.Parallel(num_threads=4, max_errors=len(exec_pairs), provide_traceback=True)
    results = parallelizer(exec_pairs)
    df['correctness_score']= [ r.correctness_score if r is not None else float(0.0) for r in results]
    return df