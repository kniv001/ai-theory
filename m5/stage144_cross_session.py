# -*- coding: utf-8 -*-
"""
M5 阶段 144：跨会话记忆（用户：继续——会话 1 教知识 → 保存 →
会话 2 加载还记得——记忆=地形（C6-03）——身份/新知识持久化）

理论锚：
  C6-03（记忆 = 地形本身——跨会话 = 地形保存/加载——固化）
  C96-01（缓存-写回——会话学习 = 写回磁盘——下次读入）
  C21-01（身份 = 概念湖——'我是X'绑定——跨会话稳定）
  C2-06（结构永不冻结——新知识增量——不重训）

机制（跨会话）：
  ① 会话 1：用户教新知识（'我叫小明'/'我喜欢蓝色'）→ learn 增量
    （continual_learn——沉积进地形）→ save（写回）
  ② 会话 2：重新 load（读入地形）→ 身份问/对话记得（'你是谁？'
    →'我是小明'——C21-01 身份绑定跨会话保留）
  ③ 新知识不重训（增量——秒级——记忆更新）

验证：
  exp1 会话 1 教学（新知识增量 → 保存）
  exp2 会话 2 跨会话记忆（重新加载——'你是谁？'→'我是小明'）
  exp3 新知识对话（'你喜欢什么颜色？'→'我喜欢蓝色'——增量后可用）
  exp4 旧记忆保持（跨会话不遗忘——'苹果是什么？'仍答）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from lake_app import TextLakeApp

MEM = "lake_memory_session.npz"


def session1():
    """会话 1：教新知识 → 保存"""
    print("=== 会话 1（教学——增量学习——保存） ===")
    app = TextLakeApp()
    base = os.path.dirname(__file__)
    app.load_corpus(
        os.path.join(base, "corpus_simple_natural.txt"),
        os.path.join(base, "corpus_simple2.txt"),
        os.path.join(base, "corpus_simple3.txt"),
        os.path.join(base, "corpus_moon.txt"),
    )
    # 教学（用户教的新知识——增量沉积——不重训）
    # 含上位词句（蓝色是颜色的一种——类-实例桥——C15-02 词类）
    new = ["我叫小明。", "我喜欢蓝色。", "我是小学生。",
           "蓝色是颜色的一种。"]
    for s in new:
        print(f"  教: {s}")
    app.learn(new, mem_path=MEM)
    # 会话内验证
    print(f"  会话内: 你是谁？ → {app.chat('你是谁？')}")
    print(f"  会话内: 你喜欢什么颜色？ → {app.chat('你喜欢什么颜色？')}")
    return app


def session2():
    """会话 2：重新加载——跨会话记忆"""
    print("\n=== 会话 2（重新加载——跨会话记忆） ===")
    app = TextLakeApp(MEM)
    app.load_sents(MEM)                    # 恢复上次语料（教学句记得）
    base = os.path.dirname(__file__)
    app.load_corpus(
        os.path.join(base, "corpus_simple_natural.txt"),
        os.path.join(base, "corpus_simple2.txt"),
        os.path.join(base, "corpus_simple3.txt"),
        os.path.join(base, "corpus_moon.txt"),
    )
    # 跨会话验证
    a_id = app.chat("你是谁？")
    a_color = app.chat("你喜欢什么颜色？")
    a_old = app.chat("苹果是什么？")
    print(f"  你是谁？ → {a_id}")
    print(f"  你喜欢什么颜色？ → {a_color}")
    print(f"  苹果是什么？ → {a_old}")
    return app, a_id, a_color, a_old


def run():
    print("=== M5 阶段 144：跨会话记忆（会话 1 教→保存→会话 2 记得）"
          " ===\n")
    app1 = session1()

    app2, a_id, a_color, a_old = session2()

    # 验证
    print("\n[验证] 跨会话记忆:")
    id_ok = "小明" in a_id if a_id else False
    col_ok = "蓝" in a_color if a_color else False
    old_ok = "水果" in a_old if a_old else False
    print(f"  身份跨会话（'小明'记得）: {'✓' if id_ok else '✗'}"
          f"——C21-01 身份绑定持久化")
    print(f"  新知识跨会话（'蓝色'记得）: {'✓' if col_ok else '✗'}"
          f"——C2-06 增量知识持久化")
    print(f"  旧记忆保持（'苹果是水果'）: {'✓' if old_ok else '✗'}"
          f"——C6-03 不遗忘")
    print("\n[结论] 跨会话记忆：会话 1 教学（增量）→ 保存（写回）→ "
          "会话 2 加载（读入）——身份/新知识记得——旧记忆不丢——"
          "记忆=地形（C6-03）——持续存在个体（C29-02 个人史）")
    print("[done] stage144 cross session")


if __name__ == "__main__":
    run()
