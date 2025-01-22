import argparse
import collections
import json
import pathlib

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/sudoku.val.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    eval_data = collections.defaultdict(list)
    with pathlib.Path(args.dataset).open() as f:
        for line in f.readlines():
            entry = json.loads(line)
            eval_data[entry["complexity"]].append(entry)

    eval_results = {}
    for complexity, entries in eval_data.items():
        print(f"Complexity: {complexity}")
        correct = 0
        total = 0
        for entry in entries:
            inputs = tokenizer.encode(entry["problem"], return_tensors="pt").to("cuda")
            outputs = model.generate(
                inputs,
                max_new_tokens=4000,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
            )
            print(tokenizer.decode(outputs[0]))
            print(entry["solution"])
            if input("good? [y/n]") == "y":
                correct += 1
            total += 1
        eval_results[complexity] = correct / total

    print(eval_results)


if __name__ == "__main__":
    main()
