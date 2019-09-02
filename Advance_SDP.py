from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP

# 主语词性集合
sub_set = {'NN', 'PN'}
# 谓语词性集合
pre_set = {'P', 'VV', 'VC'} #P 介词 VV 一般动词 VC 系动词
# 宾语词性集合
obj_set = {'NN','CC'}


# 将树结构数据转化为仅包含叶节点数据的list
def tree_flat_list(tree):
    t_list = []
    # 递归函数，获取叶子节点的label，词语，和层级
    def rec_func(tree, lv):
        for child in tree:
            # tree的结构是Tree类的多层一个嵌套，类型是Tree，只有到叶子节点的时候类型是str
            # 所以如果当前节点的下一层是str(词语)，那么这个节点的label就是词语的词性
            if len(child) == 1 and isinstance(child[0], str):
                    t_list.append((child.label(), child[0], lv + 1))
            else:
                rec_func(child, lv + 1)
    rec_func(tree, lv=0)
    return t_list


# 从叶子节点的list中抽取三元组
def t_list_triple(t_list):
    triples = []
    # 递归函数，获取所有三元组组合
    def rec_func(t_list, sub, pre, dept):
        # 有主语和谓语的情况下，在t_list中搜索宾语
        if sub and pre:
            w_ = ""
            for i, (lb, w, lv) in enumerate(t_list):
                # if lb in obj_set and lv >= dept:
                if lb in obj_set:   # 如果当前词语符合作为宾语的条件
                    # 处理多个连续主语情况，对连续主语做拼接
                    w_ += w
                    # 如果有下一个词语且下一个词语的不符合作为宾语的条件（即不连续），或者已经到了列表末尾，停止拼接，将结果输出到triples
                    if (i + 1 < len(t_list) and t_list[i + 1][0] not in obj_set) or i + 1 == len(t_list):
                        triples.append((sub, pre, w_))
                        w_ = ""
        # 有主语但没有谓语的情况
        elif sub:
            for i, (lb, w, lv) in enumerate(t_list):
                if i+1 < len(t_list):   # 如果没有到达列表的末尾（到达列表末尾就没有继续搜索的可能了）
                    # if lb in pre_set and lv >= dept:
                    if lb in pre_set:   # 如果当前词语符合作为谓语的条件
                        rec_func(t_list[i + 1:], sub, w, lv)     # 递归，变成有主语和谓语的情况，即从列表下一个元素开始搜索宾语
        # 主谓宾都没有的情况
        else:
            w_ = ""
            for i, (lb, w, lv) in enumerate(t_list):
                if i + 1 < len(t_list):  # 如果没有到达列表的末尾（到达列表末尾就没有继续搜索的可能了）
                    # if lb in sub_set and lv >= dept:
                    if lb in sub_set:  # 如果当前词语符合作为主语的条件
                        # 处理多个连续宾语情况，对连续主语做拼接
                        w_ += w
                        # 如果有下一个词语且下一个词语的不符合作为主语的条件（即不连续），或者已经到了列表末尾，停止拼接，进行递归操作
                        if (i + 1 < len(t_list) and t_list[i + 1][0] not in sub_set) or i + 1 == len(t_list):
                            rec_func(t_list[i + 1:], w_, None, lv)  # 递归，变成有主语的情况，即从列表下一个元素开始搜索谓语
                            w_ = ""
    rec_func(t_list, None, None, 0)
    return triples


# 从sentence中抽取所有可能的三元组组合
def exact_triples(sentence):
    parse = nlp.parse(sentence)
    tree = Tree.fromstring(parse)
    t_list = tree_flat_list(tree)
    triples = t_list_triple(t_list)
    return parse, tree, t_list, triples


nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05', lang='zh')
try:
    sentence = '用户可以通过网上银行申请办理信用卡业务'
    _, tree, t_list, triples = exact_triples(sentence)
    print('### 测试 ###\nsentence: ', sentence)
    print('parse result: \n', tree)
    # tree.draw()
    print('simplified tree leaves: \n', t_list)
    print('all triples results: \n', triples, '\n')

    print('### now you can input your sentence ###')
    while True:
        sentence = input("please input a sentence: ").strip()
        _, tree, t_list, triples = exact_triples(sentence)
        print('parse result: \n', tree)
        print('simplified tree leaves: \n', t_list)
        print('all triples results: \n', triples, '\n')
except KeyboardInterrupt as e:
    print(e)
    nlp.close()
finally:
    nlp.close()
