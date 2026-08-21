# -*- coding: utf-8 -*-
"""
语料标点标注（stage94 发现——用户："标点符号的训练有吗"——全语料 0 个句号——
日常语料 0 标点——wiki 只有逗号顿号——句号问号从未训练——A0 句界的
"骤降"是零河道假象——生成无终止标记=长链漂移结构性原因）

规则（语料标注——非写死机制）：
  问句行（含 怎么/什么/哪里/谁/吗/什么时候）→ 句尾加"？"
  其余陈述行 → 句尾加"。"（已有标点结尾的行跳过）
处理：所有语料文件原位写回（标点是完整句的应有部分）
"""
import os
import re

Q_MARKS = ["怎么", "什么", "哪里", "谁", "吗", "什么时候", "几"]
FILES = ["corpus_simple_natural.txt", "corpus_simple2.txt", "corpus_simple3.txt",
         "corpus_simple4.txt", "corpus_simple5.txt", "corpus_medium.txt",
         "corpus_medium2.txt", "corpus_medium3.txt", "corpus_why.txt",
         "corpus_attr_cause.txt", "corpus_negation.txt", "corpus_social.txt",
         "corpus_proper.txt"]

def is_question(s):
    return any(m in s for m in Q_MARKS)

def main():
    base = os.path.dirname(__file__)
    for fn in FILES:
        path = os.path.join(base, fn)
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        nq = ns = nskip = nword = 0
        out = []
        for s in lines:
            s_body = s.rstrip("。！？；，、")
            if len(s_body) <= 3:     # 词块行（词汇记忆单元——非句子——无标点）
                out.append(s_body)
                nword += 1
            elif s[-1] in "。！？；，、":
                out.append(s)
                nskip += 1
            elif is_question(s):
                out.append(s + "？")
                nq += 1
            else:
                out.append(s + "。")
                ns += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print(f"{fn}: 问句+? {nq} / 陈述+. {ns} / 已有 {nskip} / 词块行 {nword}")
    print("done")

if __name__ == "__main__":
    main()
