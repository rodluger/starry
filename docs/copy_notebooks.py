import json


def copy(infile, outfile):
    with open(infile, "r") as f:
        data = json.load(f)

    for cell in data.get("cells", []):
        if "hide_input" in cell.get("metadata", {}).get("tags", []):
            cell.get("source").insert(0, "#hide_input\n")

    with open(outfile, "w") as f:
        f.write(json.dumps(data, indent=2))
