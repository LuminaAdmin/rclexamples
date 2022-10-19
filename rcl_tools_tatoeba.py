from pathlib import Path

replace_map = [
    ("\r", ""),
    ("\n", " "),
    ("(", ""),
    (")", ""),
    (":", ""),
    (";", ""),
    ('"', ""),
    ("''", "'"),
    ("”", ""),
    ("”", ""),
    ("“", ""),
    ('"', ""),
    ("- ", ""),
    (" - ", " "),
    (" – ", " "),
    ("∗", " "),
    ("Mrs.", " "),
    ("Mr.", " "),
    ("  ", " "),
]
punct = {".", "?", "!"}

def clean_tatoeba(path: str) -> str:
    """Cleans a tatoeba file for use in RCL, returns file path to the new file."""
    def clean_line(line: str) -> str:
        parts = line.split("\t")
        
        p1 = parts[1].strip()
        if p1[-1] not in punct:
            p1 = f"{p1}."
        
        p2 = parts[3].strip()
        if p2[-1] not in punct:
            p2 = f"{p2}."
        
        text = f"{p1}\t{p2}"

        for p, r in replace_map:
            text = text.replace(p, r)

        return text

    file = Path(path)
    new_file = file.parent / f"{file.stem}_cleaned{file.suffix}"
    lines = file.read_text(encoding="utf-8-sig").strip().split("\n")
    text = "\n".join([clean_line(l) for l in lines])
    new_file.write_text(text, encoding="utf8")
    return str(new_file.absolute())

if __name__ == "__main__":
    new_file = clean_tatoeba("rcl_dataset_tatoeba/Sentence pairs in English-Yoruba - 2022-10-19.tsv")
    print(f"Cleaned file saved to: {new_file}")