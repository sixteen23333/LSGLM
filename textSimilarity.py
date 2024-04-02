
# coding: utf-8
import jieba
import jieba.analyse as analyse
# import jieba.posseg as pseg
# from jieba import analyse
import numpy as np
import os
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import re
import torch
import json
import random
import math
from simhash import Simhash
import html
# 自然语言处理包
# import jieba
# import jieba.analyse
# 编辑距离包
import Levenshtein
# from simtext import similarity
'''
文本相似度的计算，基于几种常见的算法的实现
'''

#python home/Graph-Bert-master/textSimilarity.py

class LevenshteinSimilarity(object):
    """
    编辑距离
    """
    def __init__(self, content_x1, content_y2):
        self.s1 = content_x1
        self.s2 = content_y2

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 正则过滤 html 标签
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        # html 转义符实体化
        content = html.unescape(content)
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=200, withWeight=False)
        return keywords

    def main(self):
        # 去除停用词
        # jieba.analyse.set_stop_words('./files/stopwords.txt')

        # 提取关键词
        keywords1 = ', '.join(self.extract_keyword(self.s1))
        keywords2 = ', '.join(self.extract_keyword(self.s2))

        # ratio计算2个字符串的相似度，它是基于最小编辑距离
        distances = Levenshtein.ratio(keywords1, keywords2)
        return distances





class TextSimilarity(object):
    
    def __init__(self,file_a,file_b):
        '''
        初始化类行
        '''
        str_a = ''
        str_b = ''
        if not os.path.isfile(file_a):
            print(file_a,"is not file")
            return
        elif not os.path.isfile(file_b):
            print(file_b,"is not file")
            return
        else:
            with open(file_a,'r') as f:
                for line in f.readlines():
                    str_a += line.strip()
                
                f.close()
            with open(file_b,'r') as f:
                for line in f.readlines():
                    str_b += line.strip()
                
                f.close()
        
        self.str_a = str_a
        self.str_b = str_b
            
    #get LCS(longest common subsquence),DP
    def lcs(str_a, str_b):
        lensum = float(len(str_a) + len(str_b))
        #得到一个二维的数组，类似用dp[lena+1][lenb+1],并且初始化为0
        lengths = [[0 for j in range(len(str_b)+1)] for i in range(len(str_a)+1)]

        #enumerate(a)函数： 得到下标i和a[i]
        for i, x in enumerate(str_a):
            for j, y in enumerate(str_b):
                if x == y:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        #到这里已经得到最长的子序列的长度，下面从这个矩阵中就是得到最长子序列
        result = ""
        x, y = len(str_a), len(str_b)
        while x != 0 and y != 0:
            #证明最后一个字符肯定没有用到
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else: #用到的从后向前的当前一个字符
                assert str_a[x-1] == str_b[y-1] #后面语句为真，类似于if(a[x-1]==b[y-1]),执行后条件下的语句
                result = str_a[x-1] + result #注意这一句，这是一个从后向前的过程
                x -= 1
                y -= 1
                
                #和上面的代码类似
                #if str_a[x-1] == str_b[y-1]:
                #    result = str_a[x-1] + result #注意这一句，这是一个从后向前的过程
                #    x -= 1
                #    y -= 1
        longestdist = lengths[len(str_a)][len(str_b)]
        ratio = longestdist/min(len(str_a),len(str_b))
        #return {'longestdistance':longestdist, 'ratio':ratio, 'result':result}
        return ratio
        
    
    def minimumEditDistance(self,str_a,str_b):
        '''
        最小编辑距离，只有三种操作方式 替换、插入、删除
        '''
        lensum = float(len(str_a) + len(str_b))
        if len(str_a) > len(str_b): #得到最短长度的字符串
            str_a,str_b = str_b,str_a
        distances = range(len(str_a) + 1) #设置默认值
        for index2,char2 in enumerate(str_b): #str_b > str_a
            newDistances = [index2+1] #设置新的距离，用来标记
            for index1,char1 in enumerate(str_a):
                if char1 == char2: #如果相等，证明在下标index1出不用进行操作变换，最小距离跟前一个保持不变，
                    newDistances.append(distances[index1])
                else: #得到最小的变化数，
                    newDistances.append(1 + min((distances[index1],   #删除
                                                 distances[index1+1], #插入
                                                 newDistances[-1])))  #变换
            distances = newDistances #更新最小编辑距离

        mindist = distances[-1]
        ratio = (lensum - mindist)/lensum
        #return {'distance':mindist, 'ratio':ratio}
        return ratio

    def levenshteinDistance(self,str1, str2):
        '''
        编辑距离——莱文斯坦距离,计算文本的相似度
        '''
        m = len(str1)
        n = len(str2)
        lensum = float(m + n)
        d = []           
        for i in range(m+1):
            d.append([i])        
        del d[0][0]    
        for j in range(n+1):
            d[0].append(j)       
        for j in range(1,n+1):
            for i in range(1,m+1):
                if str1[i-1] == str2[j-1]:
                    d[i].insert(j,d[i-1][j-1])           
                else:
                    minimum = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+2)         
                    d[i].insert(j, minimum)
        ldist = d[-1][-1]
        ratio = (lensum - ldist)/lensum
        #return {'distance':ldist, 'ratio':ratio}
        return ratio
    


    # def splitWords(self,str_a):
    #     '''
    #     接受一个字符串作为参数，返回分词后的结果字符串(空格隔开)和集合类型
    #     '''
    #     wordsa=pseg.cut(str_a)
    #     cuta = ""
    #     seta = set()
    #     for key in wordsa:
    #         #print(key.word,key.flag)
    #         cuta += key.word + " "
    #         seta.add(key.word)
        
    #     return [cuta, seta]
    
    # def JaccardSim(self,str_a,str_b):
    #     '''
    #     Jaccard相似性系数
    #     计算sa和sb的相似度 len（sa & sb）/ len（sa | sb）
    #     '''
    #     seta = self.splitWords(str_a)[1]
    #     setb = self.splitWords(str_b)[1]
        
    #     sa_sb = 1.0 * len(seta & setb) / len(seta | setb)
        
    #     return sa_sb
    
    
    # def countIDF(self,text,topK):
    #     '''
    #     text:字符串，topK根据TF-IDF得到前topk个关键词的词频，用于计算相似度
    #     return 词频vector
    #     '''
    #     tfidf = analyse.extract_tags

    #     cipin = {} #统计分词后的词频

    #     fenci = jieba.cut(text)

    #     #记录每个词频的频率
    #     for word in fenci:
    #         if word not in cipin.keys():
    #             cipin[word] = 0
    #         cipin[word] += 1

    #     # 基于tfidf算法抽取前10个关键词，包含每个词项的权重
    #     keywords = tfidf(text,topK,withWeight=True)

    #     ans = []
    #     # keywords.count(keyword)得到keyword的词频
    #     # help(tfidf)
    #     # 输出抽取出的关键词
    #     for keyword in keywords:
    #         #print(keyword ," ",cipin[keyword[0]])
    #         ans.append(cipin[keyword[0]]) #得到前topk频繁词项的词频

    #     return ans


    def cos_sim(a,b):
        a = np.array(a)
        b = np.array(b)
        
        #return {"文本的余弦相似度:":np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))}
        return np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

    def get_word_vector(s1, s2):
        """
        :param s1: 字符串1
        :param s2: 字符串2
        :return: 返回字符串切分后的向量
        """
        # 字符串中文按字分，英文按单词，数字按空格
        regEx = re.compile('[\\W]*')
        res = re.compile(r"([\u4e00-\u9fa5])")
        p1 = regEx.split(s1.lower())
        str1_list = []
        for str in p1:
            if res.split(str) == None:
                str1_list.append(str)
            else:
                ret = res.split(str)
                for ch in ret:
                    str1_list.append(ch)
        # print(str1_list)
        p2 = regEx.split(s2.lower())
        str2_list = []
        for str in p2:
            if res.split(str) == None:
                str2_list.append(str)
            else:
                ret = res.split(str)
                for ch in ret:
                    str2_list.append(ch)
        # print(str2_list)
        list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
        list_word2 = [w for w in str2_list if len(w.strip()) > 0]  # 去掉为空的字符
        # 列出所有的词,取并集
        key_word = list(set(list_word1 + list_word2))
        # 给定形状和类型的用0填充的矩阵存储向量
        word_vector1 = np.zeros(len(key_word))
        word_vector2 = np.zeros(len(key_word))
        # 计算词频
        # 依次确定向量的每个位置的值
        for i in range(len(key_word)):
            # 遍历key_word中每个词在句子中的出现次数
            for j in range(len(list_word1)):
                if key_word[i] == list_word1[j]:
                    word_vector1[i] += 1
            for k in range(len(list_word2)):
                if key_word[i] == list_word2[k]:
                    word_vector2[i] += 1

        # 输出向量
        return word_vector1, word_vector2

    def cos_dist(vec1, vec2):
        """
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 返回两个向量的余弦相似度
        """
        dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return dist1



    def eucl_sim(a,b):
        a = np.array(a)
        b = np.array(b)
        #print(a,b)
        #print(np.sqrt((np.sum(a-b)**2)))
        #return {"文本的欧几里德相似度:":1/(1+np.sqrt((np.sum(a-b)**2)))}
        return 1/(1+np.sqrt((np.sum(a-b)**2)))


    def pers_sim(a,b):
        a = np.array(a)
        b = np.array(b)

        a = a - np.average(a)
        b = b - np.average(b)

        #print(a,b)
        #return {"文本的皮尔森相似度:":np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))}
        return np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

    def splitWordSimlaryty(self,str_a,str_b,topK = 20,sim =cos_sim):
        '''
        基于分词求相似度，默认使用cos_sim 余弦相似度,默认使用前20个最频繁词项进行计算
        '''
        #得到前topK个最频繁词项的字频向量
        vec_a = self.countIDF(str_a,topK)
        vec_b = self.countIDF(str_b,topK)
        
        return sim(vec_a,vec_b)
        


    def string_hash(self,source):  #局部哈希算法的实现
        if source == "":  
            return 0  
        else:  
            #ord()函数 return 字符的Unicode数值
            x = ord(source[0]) << 7  
            m = 1000003  #设置一个大的素数
            mask = 2 ** 128 - 1  #key值
            for c in source:  #对每一个字符基于前面计算hash
                x = ((x * m) ^ ord(c)) & mask  

            x ^= len(source) # 
            if x == -1:  #证明超过精度
                x = -2  
            x = bin(x).replace('0b', '').zfill(64)[-64:]  
            #print(source,x)  

        return str(x)


    def simhash_demo(text_a, text_b):
        """
        求两文本的相似度
        :param text_a:
        :param text_b:
        :return:
        """
        a_simhash = Simhash(text_a)
        b_simhash = Simhash(text_b)
        max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
        # 汉明距离
        distince = a_simhash.distance(b_simhash)
        print(distince)
        similar = 1 - distince / max_hashbit
        return similar




def JaccardSim(str_a,str_b):
    # '''
    # Jaccard相似性系数
    # 计算sa和sb的相似度 len（sa & sb）/ len（sa | sb）
    # '''
    # seta = self.splitWords(str_a)[1]
    # setb = self.splitWords(str_b)[1]
        
    # sa_sb = 1.0 * len(seta & setb) / len(seta | setb)
    
    temp = 0
    for i in str_a:
        if i in str_b:
            temp = temp + 1
    fenmu = len(str_a) + len(str_b) - temp  # 并集
    
    jaccard_coefficient = float(temp / fenmu)  # 交集
    #import pudb;pu.db
    return jaccard_coefficient








#CUDA_VISIBLE_DEVICES=3 python home/Graph-Bert-master/textSimilarity.py


if __name__ == '__main__':
    dataset = 'other'


    doc_content_list = []
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []


#one-hot编码
    with open('home/Graph-Bert-master/data/' + dataset + '/' +'label_new_pac_head_each500_iot2022_tfc2016.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
        #import pudb;pu.db
            doc_name_list.append(line.strip())#[0]:'0\ttrain\tslowread';len:24000
            # temp = line.split("\t")
            temp = line.split()
            # import pudb;pu.db
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())#[.....'23098\ttrain\tddossim', '23099\ttrain\tddossim']

    with open('home/Graph-Bert-master/data/' + dataset + '/' + 'data_new_pac_head_each500_iot2022_tfc2016.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
        #import pudb;pu.db
            doc_content_list.append(line.strip())#[0]:'a1 b2 c3 d4 00 02 00 04 00 00 00 00 00 00 00 00 00 00 ff ff 00 00 '
    #import pudb;pu.db

    data_ids = []
    for name in doc_name_list:
        data_id = doc_name_list.index(name)
        data_ids.append(data_id)
    data_ids_str = '\n'.join(str(index) for index in data_ids)

    shuffle_doc_words_list = []
    for id in data_ids:
        shuffle_doc_words_list.append(doc_content_list[int(id)])#数据内容
        shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    score=[]
    link=[]
    link3= []
    tokens_samples= []
    for i in range(len(shuffle_doc_words_list)):
        for j in range(len(shuffle_doc_words_list)):
            if i!=j:
                # 编辑距离
                # a = LevenshteinSimilarity(shuffle_doc_words_list[i].strip(), shuffle_doc_words_list[j].strip()).main()
                # import pudb;pu.db
                # a = TextSimilarity.lcs(shuffle_doc_words_list[i].strip(),shuffle_doc_words_list[j].strip())



                # v1, v2 = TextSimilarity.get_word_vector(shuffle_doc_words_list[i].strip(),shuffle_doc_words_list[j].strip())
                # a = TextSimilarity.cos_dist(v1, v2)

                # tokens_samples.append(shuffle_doc_words_list[i].strip())
                # tokens_samples.append(shuffle_doc_words_list[j].strip())
                # token_index = {}  # 索引
                # for sample in tokens_samples:
                #     for word in sample.split():
                #         if word not in token_index:
                #             token_index[word] = len(token_index) + 1
                #
                # results = np.zeros(shape=(len(tokens_samples), max(token_index.values()) + 1))
                # for r, sample in enumerate(tokens_samples):
                #     for _, word in list(enumerate(sample.split())):
                #         index = token_index.get(word)
                #         results[i, index] = 1
                # v1 = results[0]
                # v2 = results[1]
                # a = TextSimilarity.eucl_sim(v1,v2)




                # a = TextSimilarity.simhash_demo(v1,v2)


                a=JaccardSim(shuffle_doc_words_list[i].split(),shuffle_doc_words_list[j].split())
                # import pudb;pu.db
                score.append(a)
                link3.append(str(data_ids[i]) + ' ' + str(data_ids[j]))
    avg = round(sum(score) / len(score),2)

    for t in range(len(score)):
        if score[t] >= avg:
            link.append(link3[t])


                # if a>=0.85:
                #     link2 = str(data_ids[i]) + ' ' + str(data_ids[j])
                #     link.append(link2)
    # import pudb;pu.db
    with open('home/Graph-Bert-master/data/' + dataset + '/' + 'link_new_pac_head_each500_iot2022_tfc2016.txt', 'w') as f:
        f.write("\n".join(link))

    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:#统计每个词出现次数
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)#所有词
    vocab_size = len(vocab)

    #import pudb;pu.db
    data_all = []
    for n in range(len(shuffle_doc_words_list)):
        text = shuffle_doc_words_list[n].split()
        data = []
        for s in range(len(vocab)):

            da = []
            for z in range(len(text)):
                if vocab[s] == text[z]:
                    da.append('1')
                else:
                    da.append('0')
            data_vocab = max(da)
            data.append(data_vocab)

        data = ' '.join(data)
        # data3 = str(data_ids[n]) + ' ' + data + ' ' + doc_name_list[n].split("\t")[2]

        data3 = str(data_ids[n]) + ' ' + data + ' ' + doc_name_list[n].split()[2]
        data_all.append(data3)
        #import pudb;pu.db

    random.shuffle(data_all)
    with open('home/Graph-Bert-master/data/' + dataset + '/' + 'node_new_pac_head_each500_iot2022_tfc2016.txt', "w") as f:
        f.write("\n".join(data_all))









#python home/Graph-Bert-master/textSimilarity.py


# if __name__ == '__main__':
#     dataset = 'TFC2016'
#
#
#     doc_content_list = []
#     doc_name_list = []
#     doc_train_list = []
#     doc_test_list = []
#
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'label_new_pac_head_each500.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             # import pudb;pu.db
#             doc_name_list.append(line.strip())  # [0]:'0\ttrain\tslowread';len:24000
#             temp = line.split("\t")
#             if temp[1].find('test') != -1:
#                 doc_test_list.append(line.strip())
#             elif temp[1].find('train') != -1:
#                 doc_train_list.append(line.strip())  # [.....'23098\ttrain\tddossim', '23099\ttrain\tddossim']
#
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'data_new_binary_pac_head_each500.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             # import pudb;pu.db
#             doc_content_list.append(
#                 line.strip())  # [0]:'a1 b2 c3 d4 00 02 00 04 00 00 00 00 00 00 00 00 00 00 ff ff 00 00 '
#     # import pudb;pu.db
#
#     data_ids = []
#     for name in doc_name_list:
#         data_id = doc_name_list.index(name)
#         data_ids.append(data_id)
#     data_ids_str = '\n'.join(str(index) for index in data_ids)
#
#     shuffle_doc_words_list = []
#     for id in data_ids:
#         shuffle_doc_words_list.append(doc_content_list[int(id)])  # 数据内容
#         shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
#
#     score=[]
#     link=[]
#     for i in range(len(shuffle_doc_words_list)):
#         for j in range(len(shuffle_doc_words_list)):
#             if i!=j:
#                 #编辑距离
#                 a = LevenshteinSimilarity(shuffle_doc_words_list[i].strip(), shuffle_doc_words_list[j].strip()).main()
#                 #最小编辑距离
#                 # a = minimumEditDistance(shuffle_doc_words_list[i].strip(), shuffle_doc_words_list[j].strip())
#
#                 # import pudb;pu.db
#                 # a = JaccardSim(shuffle_doc_words_list[i].split(),shuffle_doc_words_list[j].split())
#                 score.append(a)
#                 if a>=0.85:
#                     link2 = str(data_ids[i]) + ' ' + str(data_ids[j])
#                     link.append(link2)
#
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'link_new_binary_pac_head_each500_leven0.85_cut.txt', 'w') as f:
#         f.write("\n".join(link))
#
#     # 填充统一长度
#     # len_list = []
#     # for t in range(len(shuffle_doc_words_list)):
#     #     len_list.append(len(shuffle_doc_words_list[t].split()))
#     # max_length = max(len_list)
#     # doc_new = []
#     # for l in range(len(shuffle_doc_words_list)):
#     #     length =len(shuffle_doc_words_list[l].split())
#     #     doc_words_new = shuffle_doc_words_list[l]
#     #     for h in range(max_length-length):
#     #     # if length < max_length
#     #         doc_words_new = doc_words_new + ' ' + '0'
#     #         h = h + 1
#     #     doc_new.append(doc_words_new.strip())
#
#     #裁剪统一长度
#     len_list = []
#     for t in range(len(shuffle_doc_words_list)):
#         len_list.append(len(shuffle_doc_words_list[t].split()))
#     min_length = min(len_list)
#
#     doc_new = []
#     for s in range(len(shuffle_doc_words_list)):
#         length = len(shuffle_doc_words_list[s].split())
#         doc_words_new = shuffle_doc_words_list[s].split()[:min_length]
#         doc_words_new = " ".join(doc_words_new)
#         doc_new.append(doc_words_new)
#
#
#     # import pudb;pu.db
#     data_all = []
#     for l in range(len(shuffle_doc_words_list)):
#
#         data_new = str(data_ids[l]) + ' ' + doc_new[l] + ' ' + doc_name_list[l].split("\t")[2]
#         data_all.append(data_new)
#
#     random.shuffle(data_all)
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'node_new_binary_pac_head_each500_leven0.85_cut.txt', "w") as f:
#         f.write("\n".join(data_all))



























#python home/Graph-Bert-master/textSimilarity.py


# if __name__ == '__main__':
#     dataset = 'TFC2016'
#
#
#     doc_content_list = []
#     doc_name_list = []
#     doc_train_list = []
#     doc_test_list = []
#
#
# #one-hot编码
#     with open('home/Graph-Bert-master/data/' + dataset + '/' +'label_new_each500.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#         #import pudb;pu.db
#             doc_name_list.append(line.strip())#[0]:'0\ttrain\tslowread';len:24000
#             temp = line.split("\t")
#             if temp[1].find('test') != -1:
#                 doc_test_list.append(line.strip())
#             elif temp[1].find('train') != -1:
#                 doc_train_list.append(line.strip())#[.....'23098\ttrain\tddossim', '23099\ttrain\tddossim']
#
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'data_new_pac_head_each500.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#         #import pudb;pu.db
#             doc_content_list.append(line.strip())#[0]:'a1 b2 c3 d4 00 02 00 04 00 00 00 00 00 00 00 00 00 00 ff ff 00 00 '
#     #import pudb;pu.db
#
#     data_ids = []
#     for name in doc_name_list:
#         data_id = doc_name_list.index(name)
#         data_ids.append(data_id)
#     data_ids_str = '\n'.join(str(index) for index in data_ids)
#
#     shuffle_doc_words_list = []
#     for id in data_ids:
#         shuffle_doc_words_list.append(doc_content_list[int(id)])#数据内容
#         shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
#
#     score=[]
#     link=[]
#     link3= []
#     for i in range(len(shuffle_doc_words_list)):
#         for j in range(len(shuffle_doc_words_list)):
#             if i!=j:
#                 link.append(str(data_ids[i]) + ' ' + str(data_ids[j]))
#     # import pudb;pu.db
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'link_new_pac_head_each500_link_all.txt', 'w') as f:
#         f.write("\n".join(link))
#
#     word_freq = {}
#     word_set = set()
#     for doc_words in shuffle_doc_words_list:
#         words = doc_words.split()
#         for word in words:
#             word_set.add(word)
#             if word in word_freq:#统计每个词出现次数
#                 word_freq[word] += 1
#             else:
#                 word_freq[word] = 1
#
#     vocab = list(word_set)#所有词
#     vocab_size = len(vocab)
#
#     #import pudb;pu.db
#     data_all = []
#     for n in range(len(shuffle_doc_words_list)):
#         text = shuffle_doc_words_list[n].split()
#         data = []
#         for s in range(len(vocab)):
#
#             da = []
#             for z in range(len(text)):
#                 if vocab[s] == text[z]:
#                     da.append('1')
#                 else:
#                     da.append('0')
#             data_vocab = max(da)
#             data.append(data_vocab)
#
#         data = ' '.join(data)
#         data3 = str(data_ids[n]) + ' ' + data + ' ' + doc_name_list[n].split("\t")[2]
#         data_all.append(data3)
#         #import pudb;pu.db
#
#     random.shuffle(data_all)
#     with open('home/Graph-Bert-master/data/' + dataset + '/' + 'node_new_pac_head_each500_link_all.txt', "w") as f:
#         f.write("\n".join(data_all))
























#python home/Graph-Bert-master/textSimilarity.py

    # fra_len_list=[]
    # txt_all=[]
    # for k in range(len(shuffle_doc_words_list)):
    #     text = shuffle_doc_words_list[k].split()
    #     fra_len_idx=[]
    #     for j in range(len(text)-7):
    #         txt1=text[j]+text[j+1]+text[j+2]+text[j+3]
    #         txt2=text[j+4]+text[j+5]+text[j+6]+text[j+7]
    #         m1=text[j]+text[j+1]
    #         #
    #         if txt1==txt2 and txt1 !='00000000' and m1=='0000':
    #             fra_len_idx.append([k,txt1,j])
    #     fra_len_list.append(fra_len_idx)
    
    # text_new_all=[]
    # for n in range(len(shuffle_doc_words_list)):
    #     text=shuffle_doc_words_list[n].split()
    #     idx_list=[]

    #     for m in range (len(fra_len_list[n])):
    #         idx = fra_len_list[n][m][2] + 8
    #         idx_list.append(idx)


    #     pcap_1=''.join(text[0:4])
    #     pcap_2=''.join(text[4:6])
    #     pcap_3=''.join(text[6:8])
    #     pcap_4=''.join(text[8:12])
    #     pcap_5=''.join(text[12:16])
    #     pcap_6=''.join(text[16:20])
    #     pcap_7=''.join(text[20:24])
    #     pcap_new = pcap_1 + ' ' + pcap_2 + ' ' + pcap_3 + ' ' + pcap_4 + ' ' + pcap_5 + ' ' + pcap_6 + ' ' + pcap_7
    #     txt_new = pcap_new
    #     for q in range (len(idx_list)):
    #         mak1=idx_list[q]

    #         if q == len(idx_list)-1:
    #             mak2=len(text)
    #             packet_1 = ''.join(text[mak1-16:mak1-12])
    #             packet_2 = ''.join(text[mak1-12:mak1-8])
    #             packet_3 = ''.join(text[mak1-8:mak1-4])
    #             packet_4 = ''.join(text[mak1-4:mak1])
    #             packet_new = packet_1 + ' ' + packet_2 + ' ' + packet_3 + ' ' + packet_4

    #             data = ''.join(text[mak1:mak2])
    #             data2 = re.findall(r'\w{2}', data)
    #             data_new = ' '.join(data2)                
    #             txt= packet_new + ' ' + data_new
    #         else:
    #             mak2=idx_list[q+1]
    #             packet_1 = ''.join(text[mak1-16:mak1-12])
    #             packet_2 = ''.join(text[mak1-12:mak1-8])
    #             packet_3 = ''.join(text[mak1-8:mak1-4])
    #             packet_4 = ''.join(text[mak1-4:mak1])
    #             packet_new = packet_1 + ' ' + packet_2 + ' ' + packet_3 + ' ' + packet_4

    #             data = ''.join(text[mak1:mak2-17])
    #             data2 = re.findall(r'\w{2}', data)
    #             data_new = ' '.join(data2)

    #             txt= packet_new + ' ' + data_new
    #         txt_new2 = str(data_ids[n]) + ' ' + txt_new + ' ' + txt
    #         #import pudb;pu.db
    #     txt_all.append(txt_new2)
    # random.shuffle(txt_all)
    # with open('home/Graph-Bert-master/data/' + dataset + '/' + 'node_1000.txt', "w") as f:
    #     f.write("\n".join(txt_all))

    #import pudb;pu.db
















#python home/Graph-Bert-master/textSimilarity.py



    # word_freq = {}
    # word_set = set()
    # vocab=[]
    # for doc_words in shuffle_doc_words_list:
    #     words = doc_words.split()
    #     for word in words:
    #         word_set.add(word)
    #         if word in word_freq:#统计每个词出现次数
    #             word_freq[word] += 1
    #         else:
    #             word_freq[word] = 1
    
    # word_freq=sorted(word_freq.items(),key=lambda x:x[1],reverse=False)
    
    # for tx in range(len(word_freq)-100,len(word_freq)):
    #     vocab.append(word_freq[tx][0])
    # #import pudb;pu.db
    # node_all = []

    # for x in range(len(shuffle_doc_words_list)):
    #     word = shuffle_doc_words_list[x].split()
    #     node_data = data_ids[x]
    #     for s in range (len(vocab)):
    #         for l in range (len(word)):
    #             if vocab[s] == word[l]:
    #                 node_data = str(node_data) + '1' + ' '
    #                 node_all.append(node_data)
    # import pudb;pu.db
    # with open('home/Graph-Bert-master/data/' + dataset + '/' + 'node.txt', 'w') as f:
    #     f.write("\n".join(node_all))

















#python home/Graph-Bert-master/textSimilarity.py


#     for train_name in doc_train_list:
#         train_id = doc_name_list.index(train_name)
#         train_ids.append(train_id)
# # print(train_ids)
#     random.shuffle(train_ids)

# # partial labeled data
# #train_ids = train_ids[:int(0.2 * len(train_ids))]

#     train_ids_str = '\n'.join(str(index) for index in train_ids)
#     with open('home/Graph-Bert-master/data/' + dataset + '.train.index', 'w') as f:
#         f.write(train_ids_str)

# # import pudb;pu.db
#     test_ids = []
#     for test_name in doc_test_list:#test列表 4000 test Chat
#         test_id = doc_name_list.index(test_name)#返回索引值，具体第几条
#         test_ids.append(test_id)
# # print(test_ids)
#     random.shuffle(test_ids)#打乱顺序

#     test_ids_str = '\n'.join(str(index) for index in test_ids)#转将所有test_id字符,空格为间隔连一块儿

#     ids = train_ids + test_ids







    # link_cos=[]
    # link_jac=[]
    # link_min=[]
    # link_sim=[]
    # for i in range(len(shuffle_doc_words_list)):
    #     for j in range(len(shuffle_doc_words_list)):
    #         if i!=j:
    #             sim = similarity()
    #             sim_all = sim.compute(shuffle_doc_words_list[i].split(),shuffle_doc_words_list[j].split())
    #             sim_cos=sim_all['Sim_Cosine']
    #             sim_jac=sim_all['Sim_Jaccard']
    #             sim_min=sim_all['Sim_MinEdit']
    #             sim_sim=sim_all['Sim_Simple']

    #             if sim_cos>=0.5:
    #                 link_cos.append([doc_name_list[i],doc_name_list[j]])

    #             if sim_jac>=0.5:
    #                 link_jac.append([doc_name_list[i],doc_name_list[j]])

    #             if sim_min>=0.5:
    #                 link_min.append([doc_name_list[i],doc_name_list[j]])

    #             if sim_sim>=0.5:
    #                 link_sim.append([doc_name_list[i],doc_name_list[j]])
    # str = ''.join('%s' %id1 for id1 in link_cos)
    # #import pudb;pu.db
    # with open('/home/dell/lyq/home/Graph-Bert-master/data/' + dataset + '/' + 'link_cos.txt', 'w') as f:
    #     f.write(str)



        #python home/Graph-Bert-master/textSimilarity.py
