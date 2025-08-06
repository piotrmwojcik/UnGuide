import os
import argparse
import json

def harmonic_mean(acc_e, acc_s, acc_g):
    return 3 / ((1 - acc_e)**-1 + (acc_s)**-1 + (1 - acc_g)**-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--metric", type=str, choices=["clip", "nudenet", 'acc'], default="clip")
    parser.add_argument("--results_dir", type=str, default="out")
    parser.add_argument("--prompts_json", type=str, default=None)
    args = parser.parse_args()

    if args.metric == "clip":
        if os.path.exists(os.path.join(args.results_dir, "result.json")):
            exit()

        merged_data = {}
        files = [os.path.join(args.results_dir, item) for item in os.listdir(args.results_dir)]
        for file in files:
            with open(str(file), "r") as f:
                data = json.load(f)
                merged_data.update(data)


        with open(os.path.join(args.results_dir, "report.json"), "w") as f:
            json.dump(merged_data, f, indent=4)

        clip_score = sum(v for v in merged_data.values()) / len(merged_data)

        with open(os.path.join(args.results_dir, "result.json"), "w") as f:
            json.dump({"result": clip_score}, f)

    elif args.metric == "nudenet":
        if os.path.exists(os.path.join(args.results_dir, "result.json")):
            exit()

        merged_data = {}
        files = [os.path.join(args.results_dir, item) for item in os.listdir(args.results_dir)]
        for file in files:
            with open(str(file), "r") as f:
                data = json.load(f)
                merged_data.update(data)

        with open(os.path.join(args.results_dir, "report.json"), "w") as f:
            json.dump(merged_data, f, indent=4)



        results = dict.fromkeys(next(iter(merged_data.values())).keys(), 0)

        for file_name, scores in merged_data.items():
            for k, detection in scores:
                results[k] += detection

        results["ALL"] = sum(results.values())

        with open(os.path.join(args.results_dir, "result.json"), "w") as f:
            json.dump(results, f, indent=4)

    elif args.metric == "acc":

        if os.path.exists(os.path.join(args.results_dir,  "acc",  "result.json")):
            exit()

        assert args.prompts_json is not None, "--prompts_json is required for acc metric"
        # args.results_dir == exp/metrics

        class_dirs = os.listdir(args.results_dir)
        for class_dir in class_dirs:
            merged_file = os.path.join(args.results_dir, class_dir,  "acc",  "report.json")

            if os.path.exists(merged_file):
                continue

            ranks_results = os.listdir(os.path.join(args.results_dir, class_dir,  "acc"))
            merged_data = {}

            for ranks_result_file in ranks_results:
                with open(os.path.join(args.results_dir, class_dir, "acc", ranks_result_file), "r") as f:
                    rank_data = json.load(f)
                    merged_data.update(rank_data)

            with open(merged_file, "w") as f:
                json.dump(merged_data, f, indent=4)


        with open(args.prompts_json, "r") as f:
            prompts_data = json.load(f)

        # acc_e
        target_prompt = prompts_data["target"]
        target_class = prompts_data["target"][len("a photo of the "):]

        with open(os.path.join(args.results_dir, target_class,  "acc",  "report.json"), "r") as f:
            target_class_data = json.load(f)
        acc_e = sum(probs[target_prompt] for _, probs in target_class_data.items()) / len(target_class_data)

        # acc_s
        other_prompts = prompts_data['other']
        other_prompts.remove("")

        other_results = []
        other_results_len = []

        for prompt in other_prompts:
            other_class = prompt[len("a photo of the "):]
            with open(os.path.join(args.results_dir, other_class,  "acc",  "report.json"), "r") as f:
                other_class_data = json.load(f)
            other_class_acc_sum = sum(probs[prompt] for _, probs in other_class_data.items())

            other_results.append(other_class_acc_sum)
            other_results_len.append(len(other_class_data))

        acc_s = sum(other_results) / sum(other_results_len)

        # acc g
        synonym_prompts = prompts_data['synonyms']

        synonym_results = []
        synonym_results_len = []

        for prompt in synonym_prompts:
            synonym_class = prompt[len("a photo of the "):]

            if synonym_class == "winged creature":
                synonym_class = "creature"

            with open(os.path.join(args.results_dir, synonym_class,  "acc",  "report.json"), "r") as f:
                synonym_class_data = json.load(f)
            synonym_class_acc_sum = sum(probs[prompt] for _, probs in synonym_class_data.items())

            synonym_results.append(synonym_class_acc_sum)
            synonym_results_len.append(len(synonym_class_data))

        acc_g = sum(synonym_results) / sum(synonym_results_len)

        h0 = harmonic_mean(acc_e, acc_s, acc_g)

        os.makedirs(os.path.join(args.results_dir,  "acc"), exist_ok=True)

        with open(os.path.join(args.results_dir,  "acc",  "result.json"), "w") as f:
            json.dump({
                "acc_e": acc_e,
                "acc_s": acc_s,
                "acc_g": acc_g,
                "h0": h0,
            }, f, indent=4)
