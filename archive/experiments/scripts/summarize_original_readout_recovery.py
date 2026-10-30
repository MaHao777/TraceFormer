"""汇总三组原读出恢复；不将冻结表示实验写成整网结构重训。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
LABELS = {"learned": "Learned weighting", "uniform_layer2": "Uniform at layer 2",
          "no_message_layer2": "No message at layer 2"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(root):
    protocol = read(root / "protocol.json")
    report = read(root / "validation_report.json")["results"]
    verified = read(root / "independent_verification.json")
    separate = read(root / "separate_process_rescore.json")
    trained = read(root / "training_complete.json")
    evaluated = read(root / "evaluation_complete.json")
    history = read(root / "history.json")
    if (evaluated["status"] != "complete" or trained["epochs"] != protocol["epochs"]
            or trained["optimizer_steps_per_arm"] != 99 * protocol["epochs"]):
        raise ValueError("incomplete or unequal recovery budget")
    for name in LABELS:
        if not verified[name]["bit_exact"] or len(verified[name]["files"]) != 24:
            raise ValueError("incomplete independent original-graph verification")
        if verified[name]["aggregate"] != report[name]["selected"]["aggregate"]:
            raise ValueError("selected and independent global metrics differ")
        if (separate["status"] != "verified_in_separate_process"
                or separate["results"][name]["ap"] != report[name]["ap"]
                or separate["results"][name]["aggregate"] != verified[name]["aggregate"]):
            raise ValueError("separate-process rescore does not reproduce selected results")
    first = read(REPO / "outputs/original_neural_paths_20261024/report.json")["results"]
    second = read(REPO / "outputs/original_summary_weighting_20261024/report.json")["results"]
    frozen = {"learned": first["full"], "uniform_layer2": second["uniform_weights_1"],
              "no_message_layer2": first["no_message_1"]}
    frozen_gap = frozen["learned"]["ap"] - frozen["uniform_layer2"]["ap"]
    recovered_gap = report["learned"]["ap"] - report["uniform_layer2"]["ap"]
    total_seconds = trained["preparation_seconds"] + trained["seconds"] + evaluated["seconds"]
    comparisons = {}
    for left, right in (("learned", "uniform_layer2"), ("learned", "no_message_layer2"),
                        ("uniform_layer2", "no_message_layer2")):
        comparisons[f"{left}-{right}"] = {
            "ap": report[left]["ap"] - report[right]["ap"],
            "score_selected": report[left]["selected"]["aggregate"]["score"]
                              - report[right]["selected"]["aggregate"]["score"],
            "score_original_threshold": report[left]["original_threshold"]["aggregate"]["score"]
                                        - report[right]["original_threshold"]["aggregate"]["score"]}
    numbers = {"scope": protocol["scope"], "frozen": frozen, "recovered": report,
               "comparisons": comparisons, "frozen_ap_gap": frozen_gap,
               "recovered_ap_gap": recovered_gap, "compute_seconds": total_seconds,
               "cache_seconds": trained["preparation_seconds"], "training_phase_seconds": trained["seconds"],
               "evaluation_seconds": evaluated["seconds"], "peak_rss_bytes": trained["peak_rss_bytes"]}
    (root / "summary.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    colors = ("#1f77b4", "#e08214", "#557744")
    for color, name in zip(colors, LABELS, strict=True):
        axes[0].plot([row["epoch"] for row in history],
                     [row["validation_ap"][name] for row in history], "o-", label=LABELS[name], color=color)
    axes[0].set(xlabel="Common recovery epoch", ylabel="Global validation AP", title="Original readout retraining")
    axes[0].set_xticks([row["epoch"] for row in history])
    axes[0].legend(fontsize=8)
    indices = np.arange(3)
    axes[1].bar(indices - .18, [frozen[name]["ap"] for name in LABELS], .36,
                label="Frozen original readout", color="#b5bec9")
    axes[1].bar(indices + .18, [report[name]["ap"] for name in LABELS], .36,
                label="Retrained original readout", color=colors)
    axes[1].set(xticks=indices, xticklabels=["Learned", "Uniform L2", "No message L2"],
                ylim=(0, 1.1), yticks=np.arange(0, 1.01, .2), ylabel="Global validation AP",
                title="Before and after readout retraining")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Frozen representations; original decoder and refinement only; seed 37; 99 train / 24 val", fontsize=10)
    figure.savefig(root / "recovery_results.png", dpi=180)
    figure.savefig(root / "recovery_results.pdf")
    plt.close(figure)
    lines = ["# 原读出恢复对照（第2层原有消息路径）", "",
        f"三组共用99训练/24验证、seed37、{protocol['epochs']}轮，每组重训原读出的14,963参数。"
        "编码器/Tokenizer/Sparse表示冻结；无新增网络结构、无测试集访问。", "",
        "|路径|恢复前AP|恢复后AP|选中轮次|验证阈值|恢复后Score|原阈值Score|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in LABELS:
        value = report[name]
        lines.append(f"|{LABELS[name]}|{frozen[name]['ap']:.6f}|{value['ap']:.6f}|{value['epoch']}|"
            f"{value['selected']['threshold']:.6f}|{value['selected']['aggregate']['score']:.6f}|"
            f"{value['original_threshold']['aggregate']['score']:.6f}|")
    lines += ["", f"学习加权−均匀加权的AP差由恢复前{frozen_gap * 100:.3f}个百分点，"
        f"变为恢复后{recovered_gap * 100:.3f}个百分点。", "",
        "## 比较与验证", "",
        "|同预算差值|AP差|独立选阈值Score差|原固定阈值Score差|",
        "|---|---:|---:|---:|"]
    for name, values in comparisons.items():
        lines.append(f"|{name}|{values['ap']:+.6f}|{values['score_selected']:+.6f}|"
                     f"{values['score_original_threshold']:+.6f}|")
    lines += ["", "全部三组选中原读出装回原始生产神经网络，直接读取24个原始NPZ。"
        "72次原图前向logits与缓存读出逐位一致，另一计数实现逐文件与全局八项计数完全复现。"
        "另起独立进程重读24原始标注与保存logits，三组AP、全部计数、协议和源权重哈希再次精确通过。"
        "推理标签/身份清零；验证只用于AP选轮次和19+9网格阈值选择。", "",
        "## 代价", "", f"缓存{trained['preparation_seconds'] / 60:.2f}分钟；"
        f"加载/门槛/训练/逐轮验证{trained['seconds'] / 60:.2f}分钟；"
        f"最终阈值选择与原图复评{evaluated['seconds'] / 60:.2f}分钟；"
        f"上述主实验合计{total_seconds / 60:.2f}分钟。进程峰值RSS {trained['peak_rss_bytes'] / 2**30:.2f} GiB。"
        "另进程文件审计、实现检查和绘图不包含在这三个阶段计时中。", "",
        "## 解释边界", "", "这是冻结原表示后、重置原解码器和精修头的条件恢复实验。"
        "它排查原读出对分布的适应，不能排除冻结编码器后续层的协同适应，也不能当作从头整网结构消融。"
        "单种子、短程训练、重复使用开发验证集，不声称统计显著性或独立测试优势。"
        "实测三组都在第2轮继续提高AP，且学习加权组尚未恢复到原训练读出的0.925944；"
        "共同短程预算尚未证明各组收敛，因此差值也可能包含读出学习速度的差别。", "",
        "精修使用原四项主读出损失、cell辅助监督和密集序列权重；本阶段不优化Sparse分支辅助目标。"
        "三组相同初值、Dropout随机数、序列顺序和对称hard-negative query。", "",
        "本次同预算训练支持学习加权产生更容易被原读出利用的表示；恢复前的大幅差距多数可以恢复。"
        "下一步若要确认编码器结构贡献，需要让被干预层及后续编码层也得到相同的重新适应机会。"
        "当前结果不替代该确认，也不据此重写论文核心创新。", "",
        "原始证据：`outputs/original_readout_recovery_20261024/`；主报告为`validation_report.json`，"
        "独立核验为`independent_verification.json`，完整曲线为`history.json`。", ""]
    destination = REPO / "docs/original_readout_recovery_20261024.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)
    print(json.dumps(comparisons, indent=2))
    print(f"total compute {total_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root.resolve())
