import os
import joblib
import jieba
import pandas as pd

# 路径设置
base_path = 'E:\\Desktop\\Fake-news-detection-main\\Fake_news\\'
train_path = os.path.join(base_path, 'train_text')

# 加载模型和向量化器
clf = joblib.load(os.path.join(base_path, 'model.joblib'))
vectorizer = joblib.load(os.path.join(base_path, 'vectorizer.joblib'))
svd = joblib.load(os.path.join(base_path, 'svd.joblib'))
lda = joblib.load(os.path.join(base_path, 'lda.joblib'))

# 停止词
with open(os.path.join(base_path, 'stop_word.txt'), 'r', encoding='utf-8', errors='ignore') as file:
    stop_words = set(file.read().strip().split())

def segment_and_remove_stop_words(text):
    words = jieba.cut(text)
    return ' '.join(word for word in words if word not in stop_words)

def preprocess_text(text):
    # 合并文本内容
    combined_text = f"官方账户名新闻标题新闻内容{text}"
    # 分词处理
    cut_text = segment_and_remove_stop_words(combined_text)
    # TF-IDF向量化
    tfidf_vector = vectorizer.transform([cut_text])
    # SVD和LDA 特征提取
    svd_vector = svd.transform(tfidf_vector)
    lda_vector = lda.transform(tfidf_vector)
    # 合并特征
    features = pd.concat([pd.DataFrame(svd_vector), pd.DataFrame(lda_vector)], axis=1)
    return features

def predict_news(text):
    features = preprocess_text(text)
    return clf.predict_proba(features)[:, 1][0]

# 创建DataFrame用于存储数据
data = pd.DataFrame(columns=['File', 'Probability'])

# # 遍历train_text中的每一个.txt文件
# file_list = os.listdir(train_path)[:12000]  # 选择前10000个文件
# for file_name in file_list:
#     with open(os.path.join(train_path, file_name), 'r', encoding='utf-8') as file:
#         file_content = file.read()
#         probability = predict_news(file_content)
#         data = data.append({'File': file_name, 'Probability': probability}, ignore_index=True)
#
#
#
#
# # 将数据存入data.xlsx
# data.to_csv(os.path.join(base_path, 'data.csv'), index=False)

# 创建DataFrame用于存储数据
data = pd.DataFrame(columns=['File', 'Probability'])

# 遍历train_text中的每一个.txt文件并写入文件
file_list = os.listdir(train_path)[:15000]  # 选择前15000个文件
for index, file_name in enumerate(file_list, start=1):
    with open(os.path.join(train_path, file_name), 'r', encoding='utf-8') as file:
        file_content = file.read()
        probability = predict_news(file_content)
        data = data.append({'File': file_name, 'Probability': probability}, ignore_index=True)

    # 输出当前写入文件的索引
    print(f'Writing data for file {index}/{len(file_list)}')

# 将数据存入data.xlsx
data.to_csv(os.path.join(base_path, 'data.csv'), index=False)


# 通过对分类器的性能进行评估，ROC曲线的AUC值达到了0.82，表明分类器具有较好的区分能力。在ROC曲线上，通过分析找到的最佳分类阈值为0.750695636，这个值对应于使True Positive Rate (TPR)和False Positive Rate (FPR)之间差值最大的点，位于ROC曲线的最左上角。
# 当对一个文本进行分类时，如果模型预测的概率小于0.750695636，则该文本被分类为假新闻。如果模型预测的概率大于或等于0.750695636，则该文本被分类为真新闻。