"""汇总配对实验并执行预先登记的表述门槛；可等待本次队列结束。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def report(root: Path) -> dict:
    suite = read(root / "suite.json")
    results = {}
    for job in suite["jobs"]:
        done_path = root / job["id"] / "done.json"
        if not done_path.exists():
            continue
        done = read(done_path)
        checkpoint = Path(done["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != done["sha256"]:
            raise ValueError(f"checkpoint changed: {job['id']}")
        item = read(Path(done["report"]))
        if item["metrics"]["validation_files"] != 24 or set(item["per_file"]) != {
            f"val_{index:03d}" for index in range(24)
        }:
            raise ValueError("incomplete validation result")
        if item["metrics"] != read(checkpoint.parent / "reproduced_metrics.json"):
            raise ValueError("independent evaluation mismatch")
        results[job["seed"], job["result_id"]] = item
    comparisons = {}
    gate = suite["claim_gate"]
    for left, right in suite["comparisons"]:
        paired = []
        for seed in suite["seeds"]:
            if (seed, left) not in results or (seed, right) not in results:
                continue
            a, b = results[seed, left], results[seed, right]
            paired.append({
                "seed": seed,
                "score": a["metrics"]["competition"]["score"] - b["metrics"]["competition"]["score"],
                "score_at_05": a["fixed_threshold"]["competition"]["score"] - b["fixed_threshold"]["competition"]["score"],
                "ordinary_score": a["groups"]["ordinary"]["competition"]["score"] - b["groups"]["ordinary"]["competition"]["score"],
                "val023_pd": a["per_file"]["val_023"]["metrics"]["competition"]["pd"] - b["per_file"]["val_023"]["metrics"]["competition"]["pd"],
            })
        complete = len(paired) == len(suite["seeds"])
        means = {key: statistics.mean(row[key] for row in paired)
                 for key in ("score", "score_at_05", "ordinary_score", "val023_pd")} if paired else {}
        # 未完成的种子不能通过门槛；三种子通过也不等于独立测试泛化。
        passed = complete and all(row["score"] > 0 for row in paired) and (
            means["score"] >= gate["minimum_mean_score_gain"]
            and means["ordinary_score"] >= gate["minimum_mean_ordinary_group_score_gain"]
            and means["val023_pd"] >= gate["minimum_mean_val023_pd_gain"]
        )
        comparisons[f"{left}-{right}"] = {"paired": paired, "means": means,
                                         "complete": complete, "gate_passed": passed,
                                         "wording": "repeatable validation gain" if passed else "results only"}
    state = read(root / "status.json")
    output = {"queue_status": state, "completed_results": len(results),
              "expected_results": len(suite["jobs"]), "comparisons": comparisons,
              "preregistered_gate": gate, "updated_at": time.time()}
    temporary = root / "mechanism_decision.json.tmp"
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(root / "mechanism_decision.json")
    lines = ["# 核心机制验证", "", f"完成 {len(results)}/{len(suite['jobs'])} 组。",
             "验证集选择 checkpoint 与阈值；这里不检验独立测试泛化或统计显著性。", "",
             "|比较|种子|ΔScore|ΔScore@0.5|普通组 ΔScore|val_023 ΔPd|",
             "|---|---:|---:|---:|---:|---:|"]
    for name, comparison in comparisons.items():
        for row in comparison["paired"]:
            lines.append(f"|{name}|{row['seed']}|{row['score']:+.6f}|{row['score_at_05']:+.6f}|"
                         f"{row['ordinary_score']:+.6f}|{row['val023_pd']:+.6f}|")
        lines.extend(["", f"{name}：{comparison['wording']}；完成={comparison['complete']}。", ""])
    (root / "mechanism_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    arguments = parser.parse_args()
    while True:
        result = report(arguments.suite.resolve())
        print(f"{result['queue_status']['status']}: {result['completed_results']}/{result['expected_results']}", flush=True)
        if not arguments.wait or result["queue_status"]["status"] in {"complete", "failed"}:
            break
        time.sleep(60)
