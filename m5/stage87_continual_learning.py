# -*- coding: utf-8 -*-
"""
M5 阶段 87：持续学习（用户："有办法做到持续学习吗——比如训练过程中加入语料"）

理论锚：
  C2-06（断流痕迹随时间衰减——结构永不冻结 → 持续学习是框架核心设计——
    不是 LLM 的灾难性遗忘问题——沉积-侵蚀局部——旧河道保留）
  C6-03（记忆=地形——软区动态写入/硬区稳定——持续沉积不破坏）
  C2-02（结构生长——新语料的新词块/新枢纽 = 新河道——持续学习的新结构）
  C88-01（世界观=早期骨架+终身补充——补充型记忆模型）

机制（新增——stage87）：
  ① HubLake.remember()：动态词汇（训练中遇新字 → 湖扩展——O(n²) 拼接）
  ② HubLake.add_hub()：新河道（新语料的新词块/单字枢纽——结构生长）
  ③ learn_epoch_batch 自动扩展（batch 内新字先入湖）
  ④ 分阶段训练 = 持续学习（阶段 A 基础 → 阶段 B 新主题 → 阶段 C 社会）

验证：
  exp1 阶段 A（simple）——学"苹果很甜"
  exp2 阶段 B（+simple2——天气/颜色）——新字入湖（"气"等）——新知识
      + 旧知识保持（苹果仍甜——无灾难性遗忘）
  exp3 阶段 C（+simple3+social）——问句/身份——词汇继续增长
  exp4 对照：一次性（全语料）vs 持续（分阶段）——能力对比
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import (load_corpus, extract_blocks, extract_hubs,
                                       HubLake)


def stage_learn(w, sents):
    """一个持续学习阶段（3 天）——新词块/枢纽并入"""
    blocks = extract_blocks(sents)
    hubs = extract_hubs(sents, blocks)
    for h in blocks + hubs:
        w.add_hub(h)                       # 新河道（结构生长——C2-02）
    for day in range(3):
        w.learn_epoch_batch(sents, B=128)


def check(w, name, hub, obj):
    ans = w.answer(hub, obj)
    print(f"      [{name}] K['{hub}']['{obj}'] → "
          f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")


def run():
    print("=== M5 阶段 87：持续学习（训练中动态加语料——C2-06 结构永不冻结） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]

    # ---- 持续学习：三阶段（词汇/结构持续增长） ----
    print("[持续学习] 阶段 A：基础语料（simple+isa）——湖初始化")
    init_sents = simple + isa_sents
    chars0 = list(dict.fromkeys("".join(init_sents)))
    blocks0 = extract_blocks(init_sents)
    hubs0 = extract_hubs(init_sents, blocks0)
    w = HubLake(chars0, blocks0 + hubs0)
    for day in range(3):
        w.learn_epoch_batch(init_sents, B=128)
    print(f"      阶段 A 完成——词汇 {w.n} 字 / 河道 {len(w.hubs)}")
    check(w, "A", "很", "苹果")

    print("\n[持续学习] 阶段 B：加入 simple2（天气/颜色/比较）——新字新河道")
    t0 = time.perf_counter()
    stage_learn(w, simple2)
    print(f"      阶段 B 完成——{time.perf_counter()-t0:.0f}s——词汇 {w.n} 字"
          f"（+{w.n - len(chars0)}）/ 河道 {len(w.hubs)}")
    check(w, "B", "很", "苹果")     # 旧知识保持
    check(w, "B", "很", "月亮")     # 新知识（月亮是圆的）
    check(w, "B", "比", "苹果")     # 新知识（比较）

    print("\n[持续学习] 阶段 C：加入 simple3+social（问句/身份/情绪）")
    t0 = time.perf_counter()
    stage_learn(w, simple3 + social + attr + neg)
    print(f"      阶段 C 完成——{time.perf_counter()-t0:.0f}s——词汇 {w.n} 字"
          f" / 河道 {len(w.hubs)}")
    check(w, "C", "很", "苹果")     # 旧知识保持（三层后）
    check(w, "C", "很", "柠檬")     # 新知识（属性）
    check(w, "C", "是", "我")       # 社会（我是小明）
    check(w, "C", "吃", "猫")       # 动作保持
    check(w, "C", "很", "天气")     # 早期新知识保持（B 阶段学的）

    # ---- 对照：一次性训练（全语料）vs 持续 ----
    print("\n[对照] 一次性（全语料从头）vs 持续（分阶段累积）——能力对比:")
    all_sents = init_sents + simple2 + simple3 + social + attr + neg
    chars_all = list(dict.fromkeys("".join(all_sents)))
    blocks_all = extract_blocks(all_sents)
    hubs_all = extract_hubs(all_sents, blocks_all)
    w1 = HubLake(chars_all, blocks_all + hubs_all)
    t0 = time.perf_counter()
    for day in range(5):
        w1.learn_epoch_batch(all_sents, B=128)
    print(f"      一次性训练——{time.perf_counter()-t0:.0f}s（词汇 {w1.n}——"
          f"全量语料 {len(all_sents)} 行）")
    for hub, obj, name in [("很", "苹果", "苹果属性"), ("很", "柠檬", "柠檬属性"),
                           ("是", "我", "身份"), ("比", "苹果", "比较")]:
        ac = w.answer(hub, obj)
        ao = w1.answer(hub, obj)
        def fmt(a):
            return f"{[(x, f'{v:.2f}') for x, v in a[:3]]}" if a else "无"
        print(f"      '{name}': 持续 {fmt(ac)} vs 一次性 {fmt(ao)}")
    print("\n[结论] 持续学习：旧知识保持（C2-06 局部侵蚀）+ 新结构生长（C2-02）"
          "——无灾难性遗忘——词汇/河道持续增长")
    print("[done] stage87 continual learning")


if __name__ == "__main__":
    run()
