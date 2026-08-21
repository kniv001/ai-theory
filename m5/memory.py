# -*- coding: utf-8 -*-
"""
M5 记忆模块：SparseLake 持久化（保存/加载——记忆=地形 C6-03——
持续学习（增量——不重训旧语料——减少重复训练时间））

理论锚：
  C6-03（记忆 = 地形本身——硬区结构记忆/软区动态——固化 = 写回磁盘）
  C96-01（缓存-写回模型——保存 = 写回——加载 = 读入——固化）
  C2-06（结构永不冻结——持续学习——增量沉积）

机制：
  ① save_lake：K 稀疏三元组（data/indices/indptr/shape）+ chars/hubs/
    omega/gamma/t → npz（压缩）
  ② load_lake：恢复 SparseLake（不训练——直接可用）
  ③ continual_learn：增量学习（新语料只沉积新句——K += 新 csr——
    新字 remember（稀疏扩 shape）——KT 重合并——GPU 更新——
    不重训旧语料）
"""
import os
import sys
import time
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage139_wiki_sparse import SparseLake

MEMORY_FILE = "lake_memory.npz"


def save_lake(w, path=MEMORY_FILE):
    """保存记忆（C6-03——地形写回——C96-01 写回磁盘）"""
    n = w.n
    n_hub = len(w.hubs)
    # K 稀疏三元组打包（每河道 data/indices/indptr）
    datas, inds, ptrs = [], [], []
    for h in w.hubs:
        k = w.K[h]
        datas.append(k.data.astype(np.float32))
        inds.append(k.indices.astype(np.int32))
        ptrs.append(k.indptr.astype(np.int32))
    np.savez_compressed(
        path,
        chars=np.array(w.chars, dtype=object),
        hubs=np.array(w.hubs, dtype=object),
        omega=w.omega.astype(np.float32),
        gamma=w.gamma,
        t=w.t,
        n_hub=n_hub,
        k_data=np.concatenate(datas) if datas else np.array([]),
        k_indices=np.concatenate(inds) if inds else np.array([]),
        k_indptr=ptrs,                       # 每河道独立 indptr（object 数组）
        k_nnz=np.array([len(d) for d in datas], dtype=int),
        k_shape=n,
    )
    return path


def load_lake(path=MEMORY_FILE):
    """加载记忆（恢复 SparseLake——不训练——直接可用）"""
    d = np.load(path, allow_pickle=True)
    chars = [str(c) for c in d["chars"]]
    hubs = [str(h) for h in d["hubs"]]
    n = int(d["k_shape"])
    w = SparseLake.__new__(SparseLake)       # 不走 __init__（不分配空矩阵）
    w.chars = chars
    w.ci = {c: i for i, c in enumerate(chars)}
    w.hubs = hubs
    w.n = n
    w.use_gpu = torch.cuda.is_available()
    w.omega = d["omega"].astype(np.float64)
    w.gamma = float(d["gamma"])
    w.t = float(d["t"])
    # 恢复 K（csr）——注意：NpzFile 每次 __getitem__ 都重新解压成员
    # ——循环内访问会 495 次解压（~400s）——先一次性取出（解压一次）
    data_all = d["k_data"]
    idx_all = d["k_indices"]
    nnz_arr = d["k_nnz"]
    indptr_all = d["k_indptr"]
    w.K = {}
    off = 0
    for hi, h in enumerate(hubs):
        nnz = int(nnz_arr[hi])
        data = data_all[off:off + nnz]       # 内存切片（快——已解压）
        idxs = idx_all[off:off + nnz]
        off += nnz
        indptr = indptr_all[hi]
        w.K[h] = sp.csr_matrix((data, idxs, indptr), shape=(n, n))
    w.acc = {}
    # KT 合并 + GPU
    w.KT_sp = sum(w.K.values()) if w.K else sp.csr_matrix((n, n))
    if w.use_gpu:
        w.KT = torch.tensor(w.KT_sp.toarray(), device="cuda", dtype=torch.float32)
        w.rsT_gpu = torch.tensor(w.KT_sp.sum(axis=1).A1, device="cuda",
                                 dtype=torch.float32)
    else:
        w.KT = w.KT_sp.toarray()
    # 块成员索引（隔离沉积用）
    w.block_of_idx = {}
    for h in w.hubs:
        if len(h) > 1:
            for c in h:
                w.block_of_idx.setdefault(w.ci[c], []).append(h)
    return w


def continual_learn(w, new_sents, eps=0.02):
    """持续学习（增量——不重训旧语料——C2-06 结构永不冻结）：
    新句沉积 → K += 新 csr → KT 重合并 → GPU 更新"""
    # 1. 新字扩展（remember——稀疏扩 shape）
    new_chars = [c for c in dict.fromkeys("".join(new_sents))
                 if c not in w.ci]
    if new_chars:
        n_old = w.n
        n_new = n_old + len(new_chars)
        w.chars += new_chars
        for c in new_chars:
            w.ci[c] = w.n
            w.n += 1
        w.omega = np.append(w.omega, np.random.uniform(0.2, 0.6, len(new_chars)))
        # K 扩 shape（csr 重构造——shape 变大零填充——O(nnz)——
        # 切片赋值是 O(n²) 灾难（SparseEfficiencyWarning））
        # indptr 需补零（长度 = 行数+1——旧 n_old+1 → 新 w.n+1）
        for h in w.hubs:
            k = w.K[h]
            indptr_new = np.append(k.indptr,
                                   np.full(w.n - n_old, k.indptr[-1],
                                           dtype=k.indptr.dtype))
            w.K[h] = sp.csr_matrix((k.data, k.indices, indptr_new),
                                   shape=(w.n, w.n))
        print(f"  新字 {len(new_chars)}（{n_old}→{w.n}）", flush=True)
        # 块成员索引更新
        for h in w.hubs:
            if len(h) > 1:
                for c in h:
                    w.block_of_idx.setdefault(w.ci[c], []).append(h)
    # 2. 新句沉积（同 learn_v7 逻辑——只处理 new_sents）
    acc = {h: {} for h in w.hubs}
    for sent in new_sents:
        idx = [w.ci[c] for c in sent if c in w.ci]
        if len(idx) < 2:
            continue
        hit = [h for h in w.hubs if h in sent]
        if not hit:
            continue
        sub = np.array(idx)
        L = len(idx)
        di = np.arange(L)
        dist_w = 1.0 / np.maximum(np.abs(di[:, None] - di[None, :]), 1.0)
        contrib = eps * np.triu(dist_w, 1)
        pi, pj = np.nonzero(contrib)
        for p in range(len(pi)):
            i, j = int(sub[pi[p]]), int(sub[pj[p]])
            c = contrib[pi[p], pj[p]]
            bi = w.block_of_idx.get(i, [])
            bj = w.block_of_idx.get(j, [])
            shared = set(bi) & set(bj)
            for h in hit:
                if shared or not (bi or bj) or len(h) > 1:
                    a = acc[h]
                    a[(i, j)] = a.get((i, j), 0.0) + c
                    a[(j, i)] = a.get((j, i), 0.0) + c * 0.3
    # 3. K += 新 csr（增量）
    for h in w.hubs:
        a = acc[h]
        if not a:
            continue
        keys = np.array(list(a.keys()), dtype=int)
        vals = np.array(list(a.values()))
        new_csr = sp.coo_matrix((vals, (keys[:, 0], keys[:, 1])),
                                shape=(w.n, w.n)).tocsr()
        w.K[h] = w.K[h] + new_csr
    # 4. KT 重合并 + GPU 更新
    w.KT_sp = sum(w.K.values()) if w.K else sp.csr_matrix((w.n, w.n))
    if w.use_gpu:
        w.KT = torch.tensor(w.KT_sp.toarray(), device="cuda", dtype=torch.float32)
        w.rsT_gpu = torch.tensor(w.KT_sp.sum(axis=1).A1, device="cuda",
                                 dtype=torch.float32)
    else:
        w.KT = w.KT_sp.toarray()
    return w
