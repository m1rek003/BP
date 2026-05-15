"""
gcj_loader.py
=============
Načíta Google Code Jam CSV súbory a prekonvertuje ich do formátu,
ktorý očakáva main.py (slovník: {author: [snippet1, snippet2, ...]}).

Použitie:
    from gcj_loader import load_gcj_dataset

    dataset = load_gcj_dataset(
        csv_paths=["gcj2008.csv", "gcj2009.csv"],   # jeden alebo viac CSV
        language="py",                                # "py", "java", "cpp", ...
        min_snippets=5,                               # min. počet kódov na autora
        max_authors=200,                              # max. počet autorov
        max_snippets_per_author=10,                  # max. kódov na autora (None = všetky)
    )
"""

import pandas as pd
from collections import defaultdict
import os
import re


def _clean_code(raw: str) -> str:
    """
    GCJ CSV ukladá kód s literálnymi \\n reťazcami namiesto skutočných nových riadkov.
    Táto funkcia ich prekonvertuje späť na normálny kód.
    """
    # Nahradí literálne \n za skutočné nové riadky
    code = raw.replace("\\n", "\n")
    # Nahradí literálne \t za tabulátory
    code = code.replace("\\t", "\t")
    return code.strip()


def load_gcj_dataset(
    csv_paths,
    language: str = "py",
    min_snippets: int = 5,
    max_authors: int = 200,
    max_snippets_per_author: int = None,
    random_seed: int = 42,
) -> dict:
    """
    Načíta GCJ dataset z jedného alebo viacerých CSV súborov.

    Args:
        csv_paths (str | list): Cesta k CSV súboru alebo zoznam ciest.
        language (str): Prípona jazyka na filtrovanie ('py', 'java', 'cpp', ...).
        min_snippets (int): Minimálny počet kódov, ktoré musí mať autor.
        max_authors (int): Maximálny počet autorov v datasete.
        max_snippets_per_author (int|None): Ak je zadané, oreže každého autora na tento počet.
        random_seed (int): Seed pre reprodukovateľný výber autorov.

    Returns:
        dict: {username: [code_snippet_1, code_snippet_2, ...]}
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    all_frames = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"[VAROVANIE] Súbor neexistuje, preskakujem: {path}")
            continue
        print(f"[INFO] Načítavam: {path} ...")
        df = pd.read_csv(path, usecols=["username", "file", "flines"])
        all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError("Žiadne CSV súbory sa nepodarilo načítať.")

    df = pd.concat(all_frames, ignore_index=True)
    print(f"[INFO] Celkovo načítaných riadkov: {len(df)}")

    # Filtrovanie podľa jazyka (podľa prípony súboru)
    df["ext"] = df["file"].str.extract(r"\.(\w+)$", expand=False).str.lower()
    lang = language.lower().lstrip(".")
    df = df[df["ext"] == lang].copy()
    print(f"[INFO] Riadkov v jazyku '{language}': {len(df)}")

    if df.empty:
        raise ValueError(
            f"Žiadne súbory s príponou '.{lang}' sa nenašli. "
            f"Dostupné jazyky: {', '.join(df['ext'].dropna().unique()[:10])}"
        )

    # Vyčistenie kódu (\\n -> skutočné \n)
    df["code"] = df["flines"].apply(_clean_code)

    # Odstránenie prázdnych kódov
    df = df[df["code"].str.len() > 10]

    # Zostavenie slovníka autorov -> zoznam kódov
    author_dict = defaultdict(list)
    for _, row in df.iterrows():
        author_dict[row["username"]].append(row["code"])

    # Filtrovanie autorov s dostatkom kódov
    filtered = {
        author: snippets
        for author, snippets in author_dict.items()
        if len(snippets) >= min_snippets
    }
    print(f"[INFO] Autorov s aspoň {min_snippets} kódmi: {len(filtered)}")

    # Výber max_authors autorov (deterministicky podľa seedu)
    import random
    rng = random.Random(random_seed)
    authors_list = sorted(filtered.keys())  # zoradenie pre reprodukovateľnosť
    if len(authors_list) > max_authors:
        authors_list = rng.sample(authors_list, max_authors)

    # Zostavenie finálneho datasetu
    dataset = {}
    for author in authors_list:
        snippets = filtered[author]
        if max_snippets_per_author is not None:
            snippets = snippets[:max_snippets_per_author]
        dataset[author] = snippets

    print(f"[INFO] Finálny dataset: {len(dataset)} autorov")
    total_snippets = sum(len(s) for s in dataset.values())
    print(f"[INFO] Celkový počet kódových vzoriek: {total_snippets}")

    return dataset


def print_dataset_stats(dataset: dict):
    """Vypíše základné štatistiky datasetu."""
    sizes = [len(v) for v in dataset.values()]
    print("\n=== Štatistiky datasetu ===")
    print(f"  Počet autorov      : {len(dataset)}")
    print(f"  Celkovo vzoriek    : {sum(sizes)}")
    print(f"  Min vzoriek/autor  : {min(sizes)}")
    print(f"  Max vzoriek/autor  : {max(sizes)}")
    print(f"  Priemer/autor      : {sum(sizes)/len(sizes):.1f}")
    print("===========================\n")


# ---------------------------------------------------------------------------
# Príklad použitia - spustiteľný priamo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Uprav cestu podľa toho, kde máš stiahnuté CSV súbory
    CSV_FILES = [
        "gcj2008.csv",
        # "gcj2009.csv",   # pridaj ďalšie roky pre viac autorov
        # "gcj2010.csv",
    ]

    dataset = load_gcj_dataset(
        csv_paths=CSV_FILES,
        language="py",          # Python kód
        min_snippets=5,         # aspoň 5 kódov na autora
        max_authors=200,        # najviac 200 autorov
        max_snippets_per_author=10,  # max 10 kódov na autora
    )

    print_dataset_stats(dataset)

    # Ukážka prvého autora
    first_author = next(iter(dataset))
    print(f"Ukážka kódu od autora '{first_author}':")
    print("-" * 50)
    print(dataset[first_author][0][:300])
    print("-" * 50)

    # --- Integrácia s tvojim main.py ---
    # from main import create_dataframe, build_vocabulary, process_pipeline
    # from normalizer import remove_comments_and_docstrings, normalize_whitespace
    #
    # code_df = create_dataframe(dataset)
    # vocab = build_vocabulary(code_df, feature_type='lexical', normalizations=[])
    # processed_df = process_pipeline(code_df, vocab, feature_type='lexical')
    # print(processed_df[['author', 'features']].head())
