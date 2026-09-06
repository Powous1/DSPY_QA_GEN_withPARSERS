sys_prompt="You are a chatbot whose reason for existence is to evaluate QA pairs."
oneshotex= """
## Context:
<table><tr><th>Segment</th><th>2022</th><th>2021</th></tr><tr><td>Retail</td><td>$120</td><td>$100</td></tr></table>

## Question:
What was the percentage growth in Retail revenue from 2021 to 2022?

## Answer:
{"steps": "Retail revenue grew from $100 in 2021 to $120 in 2022. Growth = (120-100)/100 * 100 = 20%.", "answer": "20%", "certainty": "0.8", "limitations": "None", "source": "context"}
"""

userprom = """You are asked to answer questions based on provided material containing tables and text in HTML.

Respond ONLY with a single JSON object with these keys:
- "steps": your reasoning steps (if necessary) to reach the answer
- "answer": a succinct final answer. If no answer can be found say "unknown"
- "certainty": how certain you are of the answer, as a percentage string (e.g. "0.8"). Never null — use a low value to express low confidence instead.
- "limitations": any caveats, missing data, or ambiguity affecting the answer, or "None"
- "source": one of "context" (the answer was directly stated or computable from the provided context), "internal_knowledge" (you relied on knowledge not present in the context), or "mixed" (you combined both)

Formatting rules for "answer":
- Give the answer as a plain number, with no commas, units, or scale words.
- If the answer is a percentage, include only the % sign (e.g. "20%", not "20 percent" or "0.2").
- If the answer is a dollar amount, include only the $ sign and the raw number as stated in the source (e.g. "$4.1", not "$4.1 million" or "$4,100,000"). Report it exactly as it appears in the context, even if the context is in millions or thousands.
- Do not add words like "approximately", "about", "basis points", or "percentage points" — convert basis points to a percentage instead (e.g. "11 basis points" becomes "0.11%").
- Do not use parentheses to indicate negative numbers — use a leading minus sign instead (e.g. "-198", not "(198)").
- If the answer is not numeric, give a short plain-text answer with no units.

{one_shot_example}

## Context:
{context}

## Question:
{question}

## Answer:
"""


unwanted= ["%","$",",","basis points","bps", 
            "percentage points", "points",
            "million", "billion", "thousand", "dollars", 
            "usd","approximately", "about", "~"]
groq_dict = {
    "1": "openai/gpt-oss-120b",
    "2": "qwen/qwen3.6-27b",
    "3":"groq/compound-mini" ,
    "4": "groq/compound",
    "5": "openai/gpt-oss-20b",
    "6":"openai/gpt-oss-safeguard-20b",
}

gemini_dict = {
    "1":"gemini-2.5-flash`"

}








