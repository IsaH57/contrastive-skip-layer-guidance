import os
import json
import torch
import argparse
from glob import glob
from tqdm import tqdm
from hpsv3 import HPSv3RewardInferencer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CFG vs MSG images using HPSv3 reward model"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing 0000/, 0001/, ... subfolders"
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="comparison_results.json",
        help="Path to output JSON file"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for HPSv3 inference (VRAM dependent)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )

    return parser.parse_args()


def run_evaluation(args):
    print("Loading HPSv3 shards...")
    inferencer = HPSv3RewardInferencer(device=args.device)

    subdirs = sorted(glob(os.path.join(args.base_dir, "*/")))

    results = []
    msg_wins = 0
    cfg_wins = 0
    all_tasks = []

    print(f"Scanning {len(subdirs)} directories...")

    for subdir in tqdm(subdirs):
        meta_path = os.path.join(subdir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            prompt = json.load(f).get("positive", "")

        cfg_files = glob(os.path.join(subdir, "cfg", "*.jpg"))

        for cfg_path in cfg_files:
            filename = os.path.basename(cfg_path)
            msg_path = os.path.join(subdir, "msg", filename)

            if os.path.exists(msg_path):
                all_tasks.append({
                    "prompt": prompt,
                    "cfg_path": cfg_path,
                    "msg_path": msg_path,
                    "id": f"{os.path.basename(subdir.rstrip('/'))}_{filename}"
                })

    print(f"Total pairs found: {len(all_tasks)}. Starting inference...")

    for i in tqdm(range(0, len(all_tasks), args.batch_size)):
        batch = all_tasks[i:i + args.batch_size]

        prompts_list = []
        paths_l = []

        for task in batch:
            prompts_list.extend([task["prompt"], task["prompt"]])
            paths_l.extend([task["cfg_path"], task["msg_path"]])

        with torch.no_grad():
            rewards = inferencer.reward(
                prompts=prompts_list,
                image_paths=paths_l
            )
            scores = [r[0].item() for r in rewards]

        for j, task in enumerate(batch):
            cfg_score = scores[j * 2]
            msg_score = scores[j * 2 + 1]

            winner = "msg" if msg_score > cfg_score else "cfg"
            if winner == "msg":
                msg_wins += 1
            else:
                cfg_wins += 1

            results.append({
                "id": task["id"],
                "prompt": task["prompt"],
                "cfg_score": cfg_score,
                "msg_score": msg_score,
                "winner": winner,
                "delta": msg_score - cfg_score
            })

    final_data = {
        "summary": {
            "total_pairs": len(all_tasks),
            "msg_wins": msg_wins,
            "cfg_wins": cfg_wins,
            "msg_win_rate": (msg_wins / len(all_tasks)) * 100 if all_tasks else 0,
            "avg_delta": sum(r["delta"] for r in results) / len(results) if results else 0
        },
        "individual_results": results
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(final_data, f, indent=4)

    print("\n--- FINAL EVALUATION ---")
    print(f"Total Samples: {len(all_tasks)}")
    print(f"MSG Win Rate: {final_data['summary']['msg_win_rate']:.2f}%")
    print(f"Overall Winner: {'MSG' if msg_wins > cfg_wins else 'CFG'}")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
