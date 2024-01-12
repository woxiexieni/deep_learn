import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
def data_loader():
    # 加载数据
    data = pd.read_csv(r'D:\pythonProject\deep_learn\class_design\data\train.csv')

    # 假设您的文本数据列名为 'text_column'
    text_columns = [ 'PassengerId',	'Survived',	'Pclass','Name',	'Sex',	'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked']
    # 将多个文本列的内容合并到一个字符串列中
    data['combined_text'] = data[text_columns].astype(str).agg(' '.join, axis=1)

    # 将合并的文本列转换为单词列表
    sentences = [str(sentence).split() for sentence in data['combined_text']]
    sentences_train, sentences_test = train_test_split(sentences, test_size=0.2, random_state=42)

    # 训练 Word2Vec 模型
    embedding_dim_textcnn = 100  # 假设词向量维度为 100
    word2vec_model = Word2Vec(sentences=sentences_train, vector_size=embedding_dim_textcnn, window=5, min_count=1, workers=4)

    # 保存 Word2Vec 模型
    word2vec_model.save('word2vec_model.bin')

    # 加载 Word2Vec 模型
    word2vec_model = Word2Vec.load('word2vec_model.bin')

    # 获取词汇表和对应的词向量矩阵
    word_index_textcnn = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
    embedding_matrix_textcnn = word2vec_model.wv.vectors
    vocab_size_textcnn, embedding_dim_textcnn = embedding_matrix_textcnn.shape
    return embedding_matrix_textcnn
data = data_loader()
