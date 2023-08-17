# *_*coding:utf-8 *_*
# @Time    : 2023/6/18 19:33
# @Author  : XieSJ
# @FileName: main.py
# @Description:
import jieba
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def stopwords():
    with open("停用词表.txt", 'r', encoding='utf-8') as f:
        stopwords = [i.strip() for i in f.readlines()]
    return stopwords


# 分词函数
def tokenize(text):
    tokens = jieba.lcut(text)
    return tokens


if __name__ == '__main__':
    stopwords = stopwords()
    input_data = []
    filtered_tokens=[]
    data = pd.read_excel("副本删除后需求样本.xlsx")
    for idx, datum in data.iterrows():
        sens = [i for i in datum['sentence'].split('\n') if i!='']
        input_data.extend(sens)
    # 分词+去停用词
    for input_datum in input_data:
        tokens = tokenize(input_datum)
        # 去除停用词
        filtered_tokens.append([token for token in tokens if token not in stopwords])
    # 将处理后的文本转换为TF-IDF特征向量
    corpus = [" ".join(filtered_token) for filtered_token in filtered_tokens]  # 转换为以空格分隔的字符串
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # 使用肘部法确定最佳聚类数目
    max_clusters = 10
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(tfidf_matrix)
        inertias.append(kmeans.inertia_)
    # 绘制肘部法曲线
    plt.plot(range(1, max_clusters + 1), inertias)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    # 执行 K-means 聚类
    k = 7  # 聚类数目
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(tfidf_matrix)

    # 输出每个样本所属的聚类类别
    labels = kmeans.labels_
    columns=list(set(labels))
    output_dict={column:[] for column in columns}
    for sentence,label in zip(input_data,labels):
        output_dict[label].append(sentence)
    max_length = max(len(value) for value in output_dict.values())
    # 使用 None 补全长度不足的值数组
    for key in output_dict.keys():
        if len(output_dict[key]) < max_length:
            output_dict[key].extend([''] * (max_length - len(output_dict[key])))
    output_df=pd.DataFrame(output_dict)
    output_df.to_excel("result.xlsx",index=False)
