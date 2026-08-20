# -*- coding: utf-8 -*-
"""
语料构建脚本（可复现——2026-08-20 语料丢失教训：未跟踪大文件被 checkout 清掉——
语料必须可重建）
产出：
  corpus_wiki_filtered.txt    wiki 政治过滤语料（~20k 句——HF 流式 2000 篇）
  corpus_mixed_ordered.txt    有序混合（简单 + 知识）
"""
import os
import re

def build_wiki(out="corpus_wiki_filtered.txt", n_articles=2000, n_sents=20000):
    import os as _os
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from datasets import load_dataset
    TITLE_BLACK = ['政治','军事','战争','政府','政党','主席','总统','总理','选举','外交','殖民','革命',
      '起义','条约','战役','将军','皇帝','王朝','历史','民族','宗教','法律','宪法','军队','武器','导弹',
      '间谍','恐怖','镇压','抗议','游行','示威','专政','马克思','毛泽东','蒋介石','斯大林','希特勒',
      '共产党','国民党','美国','俄罗斯','苏联','日本','韩国','朝鲜','英国','法国','德国','印度','中东',
      '叙利亚','乌克兰','伊拉克','伊朗','阿富汗','以色列','巴勒斯坦','越南','朝鲜战争','冷战','纳粹',
      '奴隶','殖民','帝国','领土','主权','边界','移民','难民','政变','暗杀','工会','罢工']
    BODY_BLACK = ['毛泽东','习近平','邓小平','江泽民','胡锦涛','温家宝','蒋介石','斯大林','列宁','马克思',
      '希特勒','特朗普','拜登','普京','金正恩','安倍','政权','专政','侵华','帝国主义','共产党','国民党']
    ds = load_dataset('fjcanyue/wikipedia-zh-cn',
                      data_files='wikipedia-zh-cn-20260501.json', split='train', streaming=True)
    out = []
    skipped = 0
    checked = 0
    for i, row in enumerate(ds):
        if checked >= n_articles:
            break
        checked += 1
        title = row.get('title', '')
        if any(b in title for b in TITLE_BLACK):
            skipped += 1
            continue
        for sent in re.split(r'[。！？!?]', row['text']):
            s = sent.strip().replace('\n', ' ')
            if not (8 <= len(s) <= 60):
                continue
            if any(b in s for b in BODY_BLACK):
                continue
            out.append(s)
        if len(out) >= n_sents:
            break
    with open(out if isinstance(out, str) else "corpus_wiki_filtered.txt", 'w', encoding='utf-8') as f:
        pass
    with open("corpus_wiki_filtered.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f"wiki: checked {checked} articles, skipped {skipped} by title, {len(out)} sentences")
    return out

def build_mixed(simple_path="corpus_simple_natural.txt", wiki_path="corpus_wiki_filtered.txt",
                out_path="corpus_mixed_ordered.txt", wiki_n=3000):
    with open(simple_path, encoding='utf-8') as f:
        simple = [l.strip() for l in f if l.strip()]
    with open(wiki_path, encoding='utf-8') as f:
        wiki = [l.strip() for l in f if l.strip()]
    mixed = simple + wiki[:wiki_n]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mixed))
    print(f"mixed: simple {len(simple)} + wiki {min(wiki_n, len(wiki))} = {len(mixed)}")

if __name__ == "__main__":
    import sys
    if "--wiki" in sys.argv:
        build_wiki()
    else:
        build_wiki()
        build_mixed()
