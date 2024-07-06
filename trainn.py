# 数据导入
import numpy as np
import pandas as pd
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# 路径设置
base_path = 'E:\\Desktop\\Fake-news-detection-main\\Fake_news\\'

# 读取数据集
train_dataset = pd.read_csv(os.path.join(base_path, 'train.csv'))
test_dataset = pd.read_csv(os.path.join(base_path, 'test.csv'))

# 读取文本文件
def read_text_files(path, num_files):
    text_data = []
    for i in range(num_files):
        with open(os.path.join(path, f"{i}.txt"), 'r', encoding='utf-8', errors='ignore') as file:
            text_data.append(file.read())
    return text_data

train_text = read_text_files(os.path.join(base_path, 'train_text'), 10587)
test_text = read_text_files(os.path.join(base_path, 'test_text'), 10141)

# 合并train和test数据
all_text = train_text + test_text

# 合并其他列并创建合并文本数据
all_titles = pd.concat([train_dataset['Title'], test_dataset['Title']])
all_official_accounts = pd.concat([train_dataset['Ofiicial Account Name'], test_dataset['Ofiicial Account Name']])
all_contents = pd.concat([train_dataset['Report Content'], test_dataset['Report Content']])
coll = [f"{oa}{title}{text}{content}" for oa, title, text, content in zip(all_official_accounts, all_titles, all_text, all_contents)]

# 分词与停用词处理
with open(os.path.join(base_path, 'stop_word.txt'), 'r', encoding='utf-8', errors='ignore') as file:
    stop_words = set(file.read().strip().split())

def segment_and_remove_stop_words(text):
    words = jieba.cut(text)
    return ' '.join(word for word in words if word not in stop_words)

cutcoll = [segment_and_remove_stop_words(text) for text in coll]

# 特征提取
vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(cutcoll)

# 降维
svd = TruncatedSVD(n_components=100, random_state=1400)
lda = LatentDirichletAllocation(n_components=10, random_state=1337, n_jobs=-1)

svd_features = svd.fit_transform(tfidf_matrix)
lda_features = lda.fit_transform(tfidf_matrix)

# 合并特征
features = pd.concat([pd.DataFrame(svd_features), pd.DataFrame(lda_features)], axis=1)

# 模型训练
clf = LogisticRegression(max_iter=200)
clf.fit(features.iloc[:train_dataset.shape[0]], train_dataset['label'])

# 保存模型和向量化器
joblib.dump(clf, os.path.join(base_path, 'model.joblib'))
joblib.dump(vectorizer, os.path.join(base_path, 'vectorizer.joblib'))
joblib.dump(svd, os.path.join(base_path, 'svd.joblib'))
joblib.dump(lda, os.path.join(base_path, 'lda.joblib'))
# 通过将这些模型和向量化器保存到文件中，可以在以后再次使用它们，而无需重新训练模型。这有助于避免重复训练模型的时间和资源消耗。

# 预测和评估
y_proba = clf.predict_proba(features.iloc[train_dataset.shape[0]:])[:, 1]
y_predict = (y_proba > 0.5).astype(int)

acc = accuracy_score(test_dataset['label'], y_predict)
prec, rec, f1, _ = precision_recall_fscore_support(test_dataset['label'], y_predict, average='binary')

print(f"Accuracy: {acc}")
print(f"Precision, Recall, F1: {prec}, {rec}, {f1}")

# ROC曲线
fpr, tpr, _ = roc_curve(test_dataset['label'], y_proba, pos_label=1)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(fpr, tpr, color='r', linestyle='-', linewidth=1.0, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='b', linestyle='--', linewidth=1.0)
plt.legend(loc='lower right')
plt.show()