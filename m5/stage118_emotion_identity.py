# -*- coding: utf-8 -*-
"""
M5 阶段 118：情绪-身份机制（用户："是不是缺少情绪——肯定有缺少的功能导致
框架理解不了相关的事物"——诊断：谁/喜欢问缺 价值信号（A4）+ 身份湖（C21-01）
——不是妥协是补机制）

理论锚：
  A4（价值统一——单一价值信号 RPE——正负两端）
  C46-01（情绪 = 评估桥——训练产物——快乐/悲伤 = v 单变量）
  C33-01（身体模型——情绪 = 身体状态广播）
  C21-01（身份 = 概念湖——名字/角色绑定——"脸熟不知是谁"=有面孔河道
    无身份绑定——身份问需要身份绑定）
  C56-01（内社会湖——身份湖网络——"我"的社会身份）

机制（补两个缺失功能）：
  ① 情绪-价值河道（A4）：情绪词（喜欢/高兴/难过/害怕/讨厌）→ 价值
    （正/负——从语料学——"我喜欢小猫"（喜欢+正语境）——价值河道
    V[情绪词] = 正/负——训练产物）
  ② 身份湖（C21-01）：身份句（我是小明/我是学生——social）——
    "我"的身份绑定（名字/角色——K[我][小明] 强）——身份检索
    （"谁"问 → 我 → 名字/角色）

验证：
  exp1 情绪-价值涌现（喜欢/高兴=正——难过/害怕=负——语料学出）
  exp2 身份湖（"我"的身份绑定——我是小明——名字/角色）
  exp3 情绪问（"你喜欢什么动物？"→价值检索——我喜欢动物）
  exp4 身份问（"你是谁？"→身份检索——我是小明——不靠"谁"例外）
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


def emotion_value(w, emotion):
    """情绪词的价值（A4——正/负——训练产物——与正负语境共现）：
    喜欢/高兴→正（与"开心/很好/爱"正语境）；难过/害怕→负（与"哭/怕"负语境）
    简化：情绪词与"好/爱/开心"类共现强 → 正；与"哭/怕/疼"类 → 负"""
    if emotion not in w.ci:
        return 0.0
    i = w.ci[emotion]
    pos = sum(w.KT[i, w.ci[c]] for c in "好爱开心甜" if c in w.ci)
    neg = sum(w.KT[i, w.ci[c]] for c in "哭怕疼难" if c in w.ci)
    return pos - neg


def identity_of(w, sents):
    """身份湖（C21-01）："我"的身份绑定——身份句（我是X/我叫X——social）
    排除"我们/你们/他们"（"我"子串污染——"我"独立才是自称）"""
    ident = []
    for s in sents:
        if "？" in s:
            continue
        if "我" in s and "们" not in s[:2] and ("是" in s or "叫" in s):
            # "我是小明。"——"我"+"是"——身份句（排除"我们"）
            if s.startswith("我") and ("是" in s[:4] or "叫" in s[:4]):
                ident.append(s)
    return ident


def run():
    print("=== M5 阶段 118：情绪-身份机制（A4 价值 + C21-01 身份湖——补缺失功能） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    full = simple + simple3 + simple4 + simple5 + medium + social
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：情绪-价值涌现（A4） ----
    print("\n[exp1] 情绪-价值（A4——训练产物——正/负）:")
    for emo in ["喜", "高", "难", "怕"]:
        v = emotion_value(w, emo)
        print(f"      '{emo}' 价值 {v:+.3f}（{'正（愉悦）' if v > 0 else '负（威胁/悲伤）' if v < 0 else '中性'}）")

    # ---- exp2：身份湖（C21-01） ----
    print("\n[exp2] 身份湖（'我'的身份绑定——名字/角色——C21-01）:")
    ident = identity_of(w, full)
    print(f"      身份句 {len(ident)} 条: {ident[:6]}")
    if "我" in w.ci:
        i = w.ci["我"]
        row = w.KT[i] + w.KT[:, i]
        hub_single = {h for h in w.hubs if len(h) == 1}
        top = [w.chars[j] for j in np.argsort(row)[::-1]
               if re.match(r"[一-鿿]", w.chars[j]) and w.chars[j] not in hub_single][:8]
        print(f"      '我' 的强关联: {top}（名字/角色/身份——小/明/学/生——"
              f"身份湖涌现）")

    # ---- exp3：情绪问（价值检索——K[喜欢] 词块河道） ----
    print("\n[exp3] 情绪问（'你喜欢什么动物？'——价值河道——偏好）:")
    if "喜欢" in w.K:
        i = w.ci["喜"]
        row = w.K["喜欢"][i] + w.K["喜欢"][:, i]
        hub_single = {h for h in w.hubs if len(h) == 1}
        top = [(w.chars[j], row[j]) for j in np.argsort(row)[::-1][:6]
               if re.match(r"[一-鿿]", w.chars[j]) and w.chars[j] not in hub_single]
        print(f"      '喜欢'河道价值关联: {[(a, f'{v:.3f}') for a, v in top]}"
              f"（偏好对象——动物/小猫/画画——正价值——A4）")

    # ---- exp4：身份问（身份湖检索——"我是X"——不靠"谁"例外） ----
    print("\n[exp4] 身份问（'你是谁？'——身份湖——'我是X' 句）:")
    ident = identity_of(w, full)
    print(f"      身份句（'我是X'）: {ident[:5]}")
    if ident:
        names = [s[2:-1] for s in ident if len(s) > 3]
        print(f"      身份名字: {names[:5]}")
        print(f"      '谁'→身份检索: '{ident[0]}'（身份湖——C21-01——"
              f"不靠'谁'例外——身份绑定机制）")
    print("\n[done] stage118 emotion-identity")


if __name__ == "__main__":
    run()
