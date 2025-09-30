import json
import sys

import requests


def get_conceptnet_relations(word, limit=20):
    results = []

    url = f"http://api.conceptnet.io/c/en/{word.replace(' ', '_')}"

    response = requests.get(url, params={"limit": limit})
    data = response.json()

    if "edges" not in data:
        return results

    for edge in data["edges"]:
        start_label = edge.get("start", {})
        end_label = edge.get("end", {})

        if start_label.get("label", "").lower() == word.lower():
            target = end_label
            if end_label.get("language", "") != "en":
                continue
        elif end_label.get("label", "").lower() == word.lower():
            target = start_label
            if start_label.get("language", "") != "en":
                continue
        else:
            continue

        results.append(target["label"])

    return results


if __name__ == "__main__":

    words = sys.stdin.read().splitlines()

    result = {word: get_conceptnet_relations(word) for word in words}

    sys.stdout.write(json.dumps(result, indent=4))
