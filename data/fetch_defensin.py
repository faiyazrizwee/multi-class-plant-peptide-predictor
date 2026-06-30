#!/usr/bin/env python3

import requests
import pandas as pd
import time
import re
from io import StringIO

# ============================================================
# SETTINGS
# ============================================================

FAMILY = "Thionins"

FAMILIES = {
    "Defensins": {
        "query": "(taxonomy_id:33090) AND (defensin*)",
        "keywords":[
            "defensin",
            "defensin-like",
            "ec-amp",
            "pdf"
        ]
    },

    "Cyclotides": {
        "query": '(taxonomy_id:33090) AND (cyclotide OR kalata OR cycloviolacin OR circulin)',
        "keywords": [
            "cyclotide",
            "kalata",
            "cycloviolacin",
            "circulin",
            "hyen",
            "vhr",
            "cyo"
        ]
    },

    "Thionins": {
        "query": "(taxonomy_id:33090) AND (thionin*)",
        "keywords": [
            "thionin",
            "gamma-thionin",
            "thionin-like",
            "plant thionin",
            "flower-specific gamma-thionin"
        ]
    },

    "Snakins": {
        "query": '(taxonomy_id:33090) AND (snakin OR "snakin-like" OR GASA)',
        "keywords":[
            "snakin",
            "snakin-like",
            "gasa"
        ]
    },

    "Hevein_like": {
        "query": "(taxonomy_id:33090) AND (hevein)",
        "keywords": [
            "hevein",
            "hevein-like"
        ]
    },

    "LTPs": {
        "query": '(taxonomy_id:33090) AND ("lipid transfer protein" OR "non-specific lipid transfer protein" OR nsLTP)',
        "keywords": [
            "lipid transfer protein",
            "non-specific lipid transfer protein",
            "nsltp",
            "ltp"
        ]
    },

    "CLE": {
        "query": '(taxonomy_id:33090) AND (CLE OR "CLAVATA3/ESR")',
        "keywords": [
            "cle",
            "clavata3",
            "esr-related",
            "cle-related"
        ]
    },

    "RALFs": {
        "query": '(taxonomy_id:33090) AND (RALF)',
        "keywords": [
            "ralf",
            "rapid alkalinization",
            "ralf-like"
        ]
    },

    "PSKs": {
        "query": '(taxonomy_id:33090) AND (phytosulfokine)',
        "keywords": [
            "phytosulfokine",
            "psk"
        ]
    },

    "RGF_GLV": {
        "query": '(taxonomy_id:33090) AND (RGF OR GLV OR GOLVEN OR "root meristem growth factor")',
        "keywords": [
            "rgf",
            "glv",
            "golven",
            "root meristem growth factor"
        ]
    },

    "CEPs": {
        "query": '(taxonomy_id:33090) AND ("C-terminally encoded peptide")',
        "keywords": [
            "c-terminally encoded peptide",
            "cep"
        ]
    },

    "PEPs": {
        "query": '(taxonomy_id:33090) AND ("plant elicitor peptide")',
        "keywords": [
            "plant elicitor peptide",
            "pep",
            "pep1",
            "pep2",
            "pep3"
        ]
    }
}

if FAMILY not in FAMILIES:
    raise ValueError(
        f"Unknown family '{FAMILY}'. "
        f"Available families: {', '.join(FAMILIES.keys())}"
    )

OUTPUT_FILE = f"{FAMILY}.csv"
QUERY = FAMILIES[FAMILY]["query"]
KEYWORDS = FAMILIES[FAMILY]["keywords"]

FIELDS = [
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "length",
    "sequence",
    "protein_existence",
    "reviewed",
    "annotation_score"
]

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

PAGE_SIZE = 500

# ============================================================
# Download all pages
# ============================================================

all_frames = []

params = {
    "query": QUERY,
    "format": "tsv",
    "fields": ",".join(FIELDS),
    "size": PAGE_SIZE
}

url = BASE_URL

page = 1

while True:

    print(f"Downloading page {page}...")

    r = requests.get(url, params=params if page == 1 else None)

    r.raise_for_status()

    if len(r.text.strip()) == 0:
        break

    df = pd.read_csv(StringIO(r.text), sep="\t")

    all_frames.append(df)

    # -------------------------------------------------------
    # Next page
    # -------------------------------------------------------

    next_url = None

    if "Link" in r.headers:

        links = r.headers["Link"]

        m = re.search(r'<(.+?)>; rel="next"', links)

        if m:
            next_url = m.group(1)

    if next_url is None:
        break

    url = next_url
    page += 1

    time.sleep(0.2)

print("\nFinished downloading.")

# ============================================================
# Merge
# ============================================================

df = pd.concat(all_frames, ignore_index=True)

print("Downloaded :", len(df))

# ============================================================
# Filter length <=100 aa
# ============================================================

df = df[df["Length"] <= 100]

print("Length <=100 :", len(df))

# ============================================================
# Remove predicted proteins
# ============================================================

# if "Protein existence" in df.columns:
#     df = df[
#         ~df["Protein existence"]
#         .astype(str)
#         .str.contains("Predicted", case=False, na=False)
#     ]

# print("After removing predicted :", len(df))


if "Protein existence" in df.columns:
    df["Is_Predicted"] = (
        df["Protein existence"]
        .astype(str)
        .str.contains("Predicted", case=False, na=False)
    )

print("Predicted proteins :", df["Is_Predicted"].sum())
print("Non-predicted proteins :", (~df["Is_Predicted"]).sum())

# ============================================================
# Remove duplicate accession
# ============================================================

df = df.drop_duplicates(subset="Sequence")

print("Unique sequences :", len(df))

# ============================================================
# Keep only defensins
# ============================================================
pattern = "|".join(
    re.escape(k)
    for k in KEYWORDS
)

search_text = (
    df["Protein names"].fillna("")
    + " "
    + df["Gene Names"].fillna("")
)

mask = search_text.str.contains(
    pattern,
    case=False,
    regex=True,
    na=False
)

df = df[mask]

# ============================================================
# Save
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:", OUTPUT_FILE)
print("Final", FAMILY, ":", len(df))