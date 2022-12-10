from pathlib import Path

def clean_tatoeba(path: str) -> str:
    """removes numeric identifier columns from tatoeba sentence pair files"""
    def clean_line(line: str) -> str:
        parts = line.split("\t")
        origin_language_part = parts[1]     
        destination_language_part = parts[3]        
        text = f"{origin_language_part}\t{destination_language_part}"
        return text

    file = Path(path)
    new_file = file.parent / f"{file.stem}_tatoeba_prepared{file.suffix}"
    lines = file.read_text(encoding="utf8").strip().split("\n")
    text = "\n".join([clean_line(l) for l in lines])
    new_file.write_text(text, encoding="utf8")
    return str(new_file.absolute())

if __name__ == "__main__":
    new_file = clean_tatoeba("rcl_dataset_tatoeba/English-Yoruba.tsv")
    print(f"Cleaned file saved to: {new_file}")