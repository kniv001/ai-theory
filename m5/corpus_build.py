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
    # 社会政治全面过滤（用户 2026-08-20：未建立对话（社会）功能——社会政治语料是噪声——
    # 聚焦自然/科学/生活/技术领域）
    TITLE_BLACK = ['政治','军事','战争','政府','政党','主席','总统','总理','选举','外交','殖民','革命',
      '起义','条约','战役','将军','皇帝','王朝','历史','民族','宗教','法律','宪法','军队','武器','导弹',
      '间谍','恐怖','镇压','抗议','游行','示威','专政','马克思','毛泽东','蒋介石','斯大林','希特勒',
      '共产党','国民党','美国','俄罗斯','苏联','日本','韩国','朝鲜','英国','法国','德国','印度','中东',
      '叙利亚','乌克兰','伊拉克','伊朗','阿富汗','以色列','巴勒斯坦','越南','朝鲜战争','冷战','纳粹',
      '奴隶','殖民','帝国','领土','主权','边界','移民','难民','政变','暗杀','工会','罢工',
      '孙中山','袁世凯','周恩来','邓小平','习近平','江泽民','胡锦涛','温家宝','朱德','彭德怀','林彪',
      '鲁迅','胡适','宋美龄','华盛顿','林肯','罗斯福','丘吉尔','拿破仑','戴高乐','特朗普','拜登','普京',
      '金正恩','安倍','默克尔','联合国','北约','欧盟','世界银行','议会','法院','监狱','制度','主义',
      '共产主义','社会主义','资本主义','马克思主义','君主制','共和制','选举','大屠杀','911',
      '人物','生平','传记','担任','任职','当选','去世','出生','逝世','纪念','战争史','政治史','社会史']
    BODY_BLACK = ['毛泽东','习近平','邓小平','江泽民','胡锦涛','温家宝','蒋介石','斯大林','列宁','马克思',
      '希特勒','特朗普','拜登','普京','金正恩','安倍','政权','专政','侵华','帝国主义','共产党','国民党',
      '孙中山','袁世凯','周恩来','朱德','彭德怀','林彪','鲁迅','胡适','宋美龄','华盛顿','林肯','罗斯福',
      '丘吉尔','拿破仑','戴高乐','默克尔','联合国','北约','欧盟','社会主义','资本主义','共产主义',
      '马克思主义','政党','议会','法院','监狱','革命','起义','政变','暗杀','选举','大屠杀',
      '总统','总理','主席','首相','国王','皇帝','女王','将军','大使','参议员','任职','当选','逝世',
      '中华人民共和国','中国','美国','日本','俄罗斯','英国','法国','德国','印度','韩国','朝鲜','苏联',
      '经济学家','政治家','军事家','外交家','思想家','活动家','领导人','当局','政府','国务院','人大',
      '政协','中共','中共','革命家','资本家','地主','军阀']
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
            if re.search(r'\d{4,}', s):          # 数字乱串（200000000000 类——年份/数量噪声）
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
