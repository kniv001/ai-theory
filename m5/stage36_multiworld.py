# -*- coding: utf-8 -*-
"""
M5 阶段 36：多世界并行训练（阶段 3 工程化起步——island model 精英迁移）

基础：stage35v（向量化世界 + 单文件快照——5 seed 28s——24 核可用）
机制：
  ① 多进程并行：每世界独立进程（24 核 → 8 世界同时演化）
  ② 精英迁移（island model——真实：生物地理学——岛屿间物种交换——防早熟收敛）：
     每 EPOCH 步同步一次——每世界报告精英（能量最高 gid 的最优基因组）——
     广播到所有其他世界——注入替换最弱细胞——跨世界基因流动
  ③ 跨世界统计：分化率/协调度/自持率/精英能量轨迹——汇总
预期：吞吐 ×NW + 精英能量超越单世界（迁移 = 多样性保持 + 有利基因扩散）
"""
import os
import time
import numpy as np
import multiprocessing as mp
from stage35v_vectorized import CoordWorldV

NWORLD = 8          # 世界数（24 核——每进程 1 核）
EPOCH = 200         # 迁移同步间隔（步）
N_EPOCHS = 8        # 总 epoch（1600 步——8/8 分化持续可见——大种群拖慢 barrier）
N_MIG = 3           # 每世界注入精英数
TOTAL = EPOCH * N_EPOCHS


def extract_elite(w):
    """精英：能量最高的 gid 中能量最高的细胞基因组（g_mat 行 + 长度 + 节点数）"""
    gids = w.arr[:, 8].astype(int)
    e = w.arr[:, 2]
    m = gids.max() + 1
    sums = np.bincount(gids, weights=e, minlength=m)
    cnts = np.bincount(gids, minlength=m)
    avg = np.where(cnts > 0, sums / np.maximum(cnts, 1), -1.0)
    best_g = int(np.argmax(avg))
    members = np.where(gids == best_g)[0]
    bi = int(members[np.argmax(e[members])])
    return (w.g_mat[bi].copy(), int(w.g_len[bi]), int(w.n_nodes[bi]))


def inject(w, migs):
    """注入迁移基因组：替换最弱细胞（能量保底 +10——入侵者落地喘息）"""
    if not migs:
        return
    n = min(len(migs), max(1, len(w.arr) // 5))
    weak = np.argsort(w.arr[:, 2])[:n]
    for i, (gm, gl, gn) in zip(weak, migs[:n]):
        w.g_mat[i] = gm
        w.g_len[i] = gl
        w.n_nodes[i] = gn
        w.arr[i, 2] = max(w.arr[i, 2], 10.0)


def world_stats(w):
    cs_ = w.arr[:, 7]
    germ = int(np.sum(cs_ > 0.7))
    soma = int(np.sum(cs_ < 0.3))
    gids = w.arr[:, 8].astype(int)
    m = gids.max() + 1
    cnts = np.bincount(gids, minlength=m)
    uni = np.where(cnts > 0)[0]
    avg_e = float(np.mean(w.arr[:, 2]))
    return germ, soma, len(w.arr), len(uni), avg_e


def worker(seed, workdir, q_out, q_in):
    w = CoordWorldV(workdir=workdir, n0=60, seed=seed)
    for ep in range(N_EPOCHS):
        w.run(EPOCH)   # 累积步进（phase_t 持续——昼夜周期连续）
        elite = extract_elite(w)
        germ, soma, n, ng, avg_e = world_stats(w)
        q_out.put((seed, ep, elite, germ, soma, n, ng, avg_e))
        mig = q_in.get()   # 阻塞等待广播（barrier）
        inject(w, mig)
        w.savesnapshot()
    q_out.put((seed, -1, None, 0, 0, 0, 0, 0.0))


def run():
    # BLAS 线程限制（8 进程 × 24 线程 = 192 线程争 24 核——严重抖动——每进程 1 线程）
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    print(f"=== M5 阶段 36：多世界并行训练（island model——{NWORLD} 世界 × {TOTAL} 步） ===\n")
    ctx = mp.get_context("spawn")
    q_out = ctx.Queue()
    q_in = ctx.Queue()
    procs = []
    for i in range(NWORLD):
        seed = 109 + i * 97
        p = ctx.Process(target=worker, args=(seed, f"stage36_w{i}", q_out, q_in))
        p.start()
        procs.append(p)
    # 主进程：同步屏障 + 广播迁移
    ep_buf = {}        # ep -> {seed: elite}
    results = {ep: [] for ep in range(N_EPOCHS)}
    done = 0
    t0 = time.time()
    while done < NWORLD:
        seed, ep, elite, germ, soma, n, ng, avg_e = q_out.get()
        if ep == -1:
            done += 1
            continue
        results[ep].append((seed, germ, soma, n, ng, avg_e))
        ep_buf.setdefault(ep, {})[seed] = elite
        if len(ep_buf[ep]) == NWORLD:
            migs = list(ep_buf[ep].values())   # 每个精英 → 所有世界
            for _ in range(NWORLD):
                q_in.put(migs)
            if ep % 5 == 0 or ep == N_EPOCHS - 1:
                line = f"ep{ep:3d}: "
                for seed, germ, soma, n, ng, avg_e in results[ep]:
                    diff = "✓" if germ > 2 and soma > 2 else "✗"
                    line += f"w{seed%10} {diff}{germ}/{soma} e{avg_e:.0f} "
                print(line)
    for p in procs:
        p.join()
    # 汇总
    print(f"\n[汇总] {NWORLD} 世界 × {TOTAL} 步——总耗时 {time.time()-t0:.0f}s "
          f"（单世界等效 {TOTAL/1500*5*NWORLD:.0f}s+——吞吐 ×{NWORLD}）")
    for ep in range(N_EPOCHS):
        rs = results[ep]
        n_diff = sum(1 for r in rs if r[1] > 2 and r[2] > 2)
        n_live = sum(1 for r in rs if r[4] > 0)
        avg_e = np.mean([r[5] for r in rs])
        n_groups = np.mean([r[4] for r in rs])
        if ep in (0, 5, 10, N_EPOCHS - 1):
            print(f"ep{ep:3d}: 分化 {n_diff}/{NWORLD} | 活世界 {n_live}/{NWORLD} | "
                  f"精英能量 {avg_e:.0f} | 群体数 {n_groups:.0f}")
    print("[done] stage36 multiworld")


if __name__ == "__main__":
    run()
