from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from PyPDF2 import PdfReader
import pymupdf as pmu
import pdfplumber as pl
import pypdfium2 as pdfium
import re

def normalize_metadata(meta: dict) -> dict:
    if not meta:
        return {}
    normalized = {}
    for k, v in meta.items():
        key = k.lstrip("/").lower()
        normalized[key] = v
    return normalized
HEADER_FOOTER_PATTERN = re.compile(r"^.+\|.+\|\s*\d+\s*$")
def isjunk(text: str) -> bool:
    return any(HEADER_FOOTER_PATTERN.
               match(line.strip()) for 
               line in text.split("\n"))


#### PYMUPDF
def openmupdf(path):
    return pmu.open(path)

def pagemupdf(file):
    return iter(file)

def metamupdf(file):
    return normalize_metadata(file.metadata or {})

def pymupdf(page):
    table_objs = page.find_tables() or ""
    full_text=page.get_text() or ""
    table_blocks = ["\n".join(" | ".join(str(c) if c else "" for c in row)
                                for row in t.extract())
                        for t in table_objs.tables]
    combined = full_text.strip()
    if table_blocks:
        combined += "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_blocks)
    return combined

#### PDFPLUMBER
def openplumber(path):
    return pl.open(path)

def pageplumber(file):
    return file.pages

def metaplumber(file):
    return normalize_metadata(file.metadata or {})

def pdfplumber(page):
    table_objs = page.extract_tables()
    if not table_objs:
        return None
    full_text = page.extract_text() 
    table_blocks = ["\n".join(" | ".join(c or "" for c in row) for row in t) for t in table_objs if t is not None]
    return full_text.strip() + "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_blocks)

#### PYPDF2

def openpdf2(path):
    return PdfReader(path)

def pagepdf2(file):
    return file.pages

def metapdf2(file):
    return normalize_metadata(file.metadata or {})

def pypdf2(page):
    return


parse_list ={
    "pdfplumber":{ 
        "open":openplumber,"pages":pageplumber,
        "meta":metaplumber,"content":pdfplumber,
        "close": lambda d: d.close()
        },
    "pypdf2":{
        "open":openpdf2,"pages":pagepdf2,
        "meta":metapdf2,"content":pypdf2,
        "close": lambda d: d.close()
        },
    "pymupdf":{
        "open":openmupdf,"pages":pagemupdf,
        "meta":metamupdf,"content":pymupdf,
        "close": None
        }
}

