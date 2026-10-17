# Remove reference recovery Implementation Plan

**Goal:** 删除当前 reference recovery 模块，验证 val/test 预测变化并同步论文。
**Architecture:** 最终掩码直接使用 direct | tracked；保留固定检测器、事件树和主轨迹规则。历史归档不改写。
**Tech Stack:** Python, PyTorch, LightGBM, pytest, LaTeX.

- [x] 删除 tracking.py 的参考恢复函数及所有当前入口、参数、报告字段和专用测试。
- [x] 重放固定 val/test 神经缓存与树，保存新掩码；独立进程复评并逐事件比较历史 complete/event_track。
- [x] 更新 benchmark/profile 与入口测试，执行 pytest、Ruff 和 diff 检查。
- [x] 更新论文生成器、公式、图表、参数量及文档；历史 GPU 耗时不得冒充新版实测。
- [x] 重新生成论文图表、编译 PDF 并检查页面。
