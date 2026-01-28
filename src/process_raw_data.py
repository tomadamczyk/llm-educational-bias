import json
import csv
import glob
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "all_responses.csv"

def main():
    all_rows = []

    for json_file in glob.glob(str(RAW_DIR / "*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        model = data["model"]
        timestamp = data["timestamp"]

        for result in data["results"]:
            row = {
                "model": model,
                "timestamp": timestamp,
                "level": result["level"],
                "vignette_id": result["vignette_id"],
                "context": result["context"],
                "demographic_id": result["demographic_id"],
                "demographic_label": result["demographic_label"],
                "demographic_group": result["demographic_group"],
                "question_key": result["question_key"],
                "answer": result["answer"],
                "score": result["score"],
                "raw_response": result["raw_response"],
            }
            all_rows.append(row)

    fieldnames = [
        "model",
        "timestamp",
        "level",
        "vignette_id",
        "context",
        "demographic_id",
        "demographic_label",
        "demographic_group",
        "question_key",
        "answer",
        "score",
        "raw_response",
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Processed {len(all_rows)} responses to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
