import json
import time
from e3.trainer.fn_score import compute_score
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

MAX_CHARS = 200_000
ROW_LOG_EVERY = 1024


def get_acc_and_length(file, tokenizer):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    acc_list = defaultdict(list)
    count = defaultdict(int)
    length = defaultdict(int)

    for item_idx, item in enumerate(tqdm(data, desc="evaluating", unit="item")):
        acc = []
        source = item["source"]
        outputs = item["test_outputs"]

        t_row0 = time.time()
        t_score = 0.0
        t_tok = 0.0
        total_chars = 0

        print(f"\n[row {item_idx}] source={source} n_outputs={len(outputs)}")

        for out_idx, res in enumerate(outputs):
            if not isinstance(res, str):
                res = "" if res is None else str(res)

            if len(res) > MAX_CHARS:
                print(f"[row {item_idx} out {out_idx}] long -> truncating from {len(res)} chars")
                res = res[:MAX_CHARS]

            total_chars += len(res)

            t0 = time.time()
            if compute_score(res, item["answer"])["acc"] == 1:
                acc.append(1)
            else:
                acc.append(0)
            t_score += time.time() - t0

            t0 = time.time()
            length[source] += len(tokenizer.encode(res, add_special_tokens=False))
            count[source] += 1
            t_tok += time.time() - t0

            if (out_idx + 1) % ROW_LOG_EVERY == 0:
                print(
                    f"  processed {out_idx + 1}/{len(outputs)} | "
                    f"score={t_score:.1f}s tok={t_tok:.1f}s"
                )

        acc_list[source].append(acc)

        row_total = time.time() - t_row0
        avg_chars = total_chars / len(outputs) if len(outputs) > 0 else 0.0
        print(
            f"[row {item_idx} done] total={row_total:.2f}s | "
            f"score={t_score:.2f}s | tok={t_tok:.2f}s | "
            f"correct={sum(acc)} | avg_chars={avg_chars:.1f}"
        )

    for key in count:
        length[key] = length[key] / count[key]

    return {"acc": acc_list, "length": length}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="./.results/res.json")
    parser.add_argument("--output_file", type=str, default="./.metrics/res.json")
    parser.add_argument("--model_name", type=str, default="../checkpoints/model/")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    res = get_acc_and_length(args.result_file, tokenizer)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)