import os
import joblib
import jieba
import pandas as pd

# 路径设置
base_path = 'E:\\Desktop\\Fake-news-detection-main\\Fake_news\\'

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



# def predict_news(text, threshold=0.249304364):
def predict_news(text, threshold=0.750695636):
    features = preprocess_text(text)

    probability = 1 - clf.predict_proba(features)[:, 1]  # 获取预测为真新闻的概率
    # probability = 1 - probability
    global xx
    xx = probability
    print(xx)

    return "新闻文本大概率为假❌" if probability <= threshold else "新闻文本大概率为真✅"


# 新文本内容
new_text = input("请输入要预测的新闻文本：")

# 预测并输出结果
result = predict_news(new_text)
print(result)