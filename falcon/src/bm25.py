import json
import logging
import math
import os
import pickle

import jieba

# 配置 Jieba 日志级别
jieba.setLogLevel(log_level=logging.INFO)


class BM25Param(object):
    def __init__(
        self,
        f,
        df,
        idf,
        length,
        avg_length,
        docs_list,
        line_length_list,
        k1=1.2,
        k2=1.0,
        b=0.75,
    ):
        """

        :param f:
        :param df:
        :param idf:
        :param length:
        :param avg_length:
        :param docs_list:
        :param line_length_list:
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param b:
        """
        self.f = f
        self.df = df
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

    def __str__(self):
        return f"k1:{self.k1}, k2:{self.k2}, b:{self.b}"


class BM25(object):
    _param_pkl = "falcon/src/bm25_param.pkl"
    _stop_words_path = "falcon/src/stop_words.txt"
    _stop_words = []

    # 新增：用于存储需要优先分词的专有名词
    _custom_words_to_add = []

    def __init__(self, docs="", custom_words=[], doc_path=None):
        self.docs = docs
        self._custom_words_to_add = custom_words  # 接收专有名词列表
        self._docs_path = doc_path
        self._add_custom_words_to_jieba()  # 优先将专有名词加入词典
        self.param: BM25Param = self._load_param()

    def _add_custom_words_to_jieba(self):
        """将专有名词添加到Jieba词典中，确保它们不会被错误切分."""
        for word in self._custom_words_to_add:
            # 使用 suggest_freq 提高词频，确保分词器识别它为一个词
            # 也可以使用 jieba.add_word(word)
            jieba.suggest_freq(word, tune=True)

    def _load_stop_words(self):
        if not os.path.exists(self._stop_words_path):
            raise Exception(
                f"system stop words: {self._stop_words_path} not found"
            )
        stop_words = []
        with open(self._stop_words_path, "r", encoding="utf8") as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _build_param(self):

        def _cal_param(reader_obj):
            f = []
            df = {}
            idf = {}
            # lines = reader_obj.readlines()
            lines = json.load(reader)
            length = len(lines)
            words_count = 0
            docs_list = []
            line_length_list = []
            for dic in lines:
                line = dic["content"]
                line = line.strip()
                if not line:
                    continue
                # 确保分词时使用了已经添加了自定义词典的Jieba实例
                words = [
                    word
                    for word in jieba.lcut(line)
                    if word and word not in self._stop_words
                ]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                f.append(tmp_dict)
                for word in tmp_dict.keys():
                    df[word] = df.get(word, 0) + 1
            for word, num in df.items():
                # IDF公式不变
                idf[word] = math.log(length - num + 0.5) - math.log(num + 0.5)
            param = BM25Param(
                f,
                df,
                idf,
                length,
                words_count / length,
                docs_list,
                line_length_list,
                k1=0.2,
                k2=0.2,
                b=0.5,
            )
            return param

        # cal
        if self.docs:
            if not os.path.exists(self.docs):
                raise Exception(f"input docs {self.docs} not found")
            # with open(self.docs, 'r', encoding='utf8') as reader:
            with open(self.docs, "r") as reader:
                param = _cal_param(reader)

        else:
            if not os.path.exists(self._docs_path):
                raise Exception(f"system docs {self._docs_path} not found")
            # with open(self._docs_path, 'r', encoding='utf8') as reader:
            with open(self._docs_path, "r") as reader:
                param = _cal_param(reader)

        with open(self._param_pkl, "wb") as writer:
            pickle.dump(param, writer)
        return param

    def _load_param(self):
        self._stop_words = self._load_stop_words()
        # 由于我们修改了词典，如果 docs 不为空或者有自定义词，建议重新构建参数以保证准确性
        should_rebuild = self.docs or self._custom_words_to_add

        if should_rebuild:
            param = self._build_param()
        else:
            if not os.path.exists(self._param_pkl):
                param = self._build_param()
            else:
                with open(self._param_pkl, "rb") as reader:
                    param = pickle.load(reader)
        return param

    def _cal_similarity(self, words, index):
        # BM25 计算公式：
        # Score = IDF * [ (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b *
        # (|D| / avgDL))) ]
        score = 0
        for word in words:
            # 1. 确保词汇存在 IDF
            if word not in self.param.idf:
                continue

            # 2. 确保词汇在当前文档中出现
            if word not in self.param.f[index]:
                continue

            # f(q, D): 词汇在当前文档中的频率
            term_frequency = self.param.f[index][word]

            # IDF: (log((N - n_q + 0.5) / (n_q + 0.5)))
            idf_value = self.param.idf[word]

            # |D| / avgDL
            length_ratio = (
                self.param.line_length_list[index] / self.param.avg_length
            )

            # 分子: f(q, D) * (k1 + 1)
            molecular = idf_value * term_frequency * (self.param.k1 + 1)

            # 分母: f(q, D) + k1 * (1 - b + b * (|D| / avgDL))
            denominator = term_frequency + self.param.k1 * (
                1 - self.param.b + self.param.b * length_ratio
            )

            score += molecular / denominator
        return score

    def cal_similarity(self, query: str):
        """相似度计算，无排序结果 :param query: 待查询结果 :return: [(doc, score), ..]"""
        words = [
            word
            for word in jieba.lcut(query)
            if word and word not in self._stop_words
        ]
        score_list = []
        for index in range(self.param.length):
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))
        return score_list

    def cal_similarity_rank(self, query: str):
        """相似度计算，排序 增加逻辑：如果文档标题/内容包含精确的查询词，给予巨大加分."""
        result = self.cal_similarity(query)

        final_result = []
        for doc, score in result:
            bonus = 0

            if f"__{query} " in doc:
                bonus = 1000

            elif f"__{query}\n" in doc:
                bonus = 1000

            elif f"__{query}(" in doc:
                bonus = 1000

            elif f" {query} " in doc:
                bonus = 500
            if score > 0:
                final_result.append((doc, score + bonus))
            else:
                final_result.append((doc, score))

        final_result.sort(key=lambda x: -x[1])
        return final_result


if __name__ == "__main__":
    target_api_name = "__bang_matmul"
    print(f"Adding custom word to Jieba: {target_api_name}")
    bm25 = BM25(custom_words=[target_api_name])

    # ------------------

    query_content = target_api_name
    result = bm25.cal_similarity_rank(query_content)

    print("\n" + "=" * 50)
    print(f"Query: {query_content}")
    print(f"Top 3 Results:")

    for doc, score in result:
        if score > 0:
            print(f"Score: {score:.4f}, Doc Snippet: {doc[:50]}...")
            print(doc)
