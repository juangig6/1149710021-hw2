#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part A: 傳統方法
包含 A-1, A-2, A-3 三個任務
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from wordcloud import WordCloud

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# A-1: TF-IDF 文本相似度計算
# ============================================================
def run_a1():
    """A-1: TF-IDF 文本相似度計算"""
    print("\nA-1: TF-IDF 文本相似度計算")
    print("="*60)

    # 測試文本
    documents = [
        "人工智慧正在改變世界，機器學習是其核心技術",
        "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
        "今天天氣很好，適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康，每天都應該保持運動習慣"
    ]

    print("測試文檔：")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")

    # 中文分詞
    def tokenize(text):
        return " ".join(jieba.cut(text))

    print("\n" + "="*50)
    print("步驟 1: 中文分詞結果")
    print("="*50)
    tokenized_docs = [tokenize(doc) for doc in documents]
    for i, doc in enumerate(tokenized_docs, 1):
        print(f"文檔 {i}: {doc}")

    # TF-IDF 計算
    print("\n" + "="*50)
    print("步驟 2: TF-IDF 計算")
    print("="*50)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

    # 取得特徵詞彙
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n詞彙表大小: {len(feature_names)} 個詞")
    # print(f"詞彙表: {', '.join(feature_names[:20])}{'...' if len(feature_names) > 20 else ''}")
    print(f"詞彙表: {', '.join(feature_names)}")

    # 計算 IDF 值
    idf_values = dict(zip(feature_names, vectorizer.idf_))

    # 顯示每個文檔的 TF-IDF 值
    print("\n" + "="*50)
    print("步驟 3: 各文檔的 TF-IDF 值詳細分析")
    print("="*50)

    for doc_idx in range(len(documents)):
        print(f"\n【文檔 {doc_idx + 1}】: {documents[doc_idx]}")
        print("-" * 50)
        
        tfidf_vector = tfidf_matrix[doc_idx].toarray()[0]
        word_scores = [(feature_names[i], tfidf_vector[i]) 
                       for i in range(len(feature_names)) 
                       if tfidf_vector[i] > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        doc_words = tokenized_docs[doc_idx].split()
        word_count = {}
        for word in doc_words:
            word_count[word] = word_count.get(word, 0) + 1
        total_words = len(doc_words)
        
        if doc_idx == 0:
            print("\n📊 Top 10 關鍵詞 (含 TF, IDF, TF-IDF 詳細計算):")
            print(f"{'排名':<4} {'詞彙':<10} {'TF':<10} {'IDF':<10} {'TF-IDF':<10}")
            print("-" * 60)
            
            for rank, (word, tfidf_score) in enumerate(word_scores[:10], 1):
                tf = word_count.get(word, 0) / total_words
                idf = idf_values[word]
                print(f"{rank:<4} {word:<10} {tf:<10.4f} {idf:<10.4f} {tfidf_score:<10.4f}")
            
            if len(word_scores) > 10:
                print(f"\n  ... (還有 {len(word_scores) - 10} 個詞)")
            
            print("\n" + "="*60)
            print("📝 計算說明:")
            print("="*60)
            print("TF (Term Frequency, 詞頻):")
            print("  公式: TF = 該詞在文檔中出現次數 / 文檔總詞數")
            print(f"  範例: '{word_scores[0][0]}' 在文檔1中出現 {word_count.get(word_scores[0][0], 0)} 次")
            print(f"       文檔1總共有 {total_words} 個詞")
            print(f"       TF = {word_count.get(word_scores[0][0], 0)}/{total_words} = {word_count.get(word_scores[0][0], 0)/total_words:.4f}")
            
            print("\nIDF (Inverse Document Frequency, 逆文檔頻率):")
            print("  公式: IDF = log(文檔總數 / 包含該詞的文檔數)")
            
            first_word = word_scores[0][0]
            doc_containing_word = sum(1 for doc in tokenized_docs if first_word in doc.split())
            print(f"  範例: '{first_word}' 出現在 {doc_containing_word} 個文檔中")
            print(f"       總共有 {len(documents)} 個文檔")
            print(f"       IDF = log({len(documents)}/{doc_containing_word}) = {idf_values[first_word]:.4f}")
            
            print("\nTF-IDF (Term Frequency-Inverse Document Frequency):")
            print("  公式: TF-IDF = TF × IDF")
            tf_first = word_count.get(first_word, 0) / total_words
            print(f"  範例: '{first_word}' 的 TF-IDF")
            print(f"       = {tf_first:.4f} × {idf_values[first_word]:.4f}")
            print(f"       = {word_scores[0][1]:.4f}")
            
            print("\n💡 解讀:")
            print("  • TF 越高 = 該詞在此文檔中越重要")
            print("  • IDF 越高 = 該詞在整個文檔集中越罕見,越有區別性")
            print("  • TF-IDF 越高 = 該詞是此文檔的關鍵特徵詞")
            print("="*60)
        else:
            print("\nTop 10 關鍵詞 (按 TF-IDF 分數排序):")
            print(f"{'排名':<4} {'詞彙':<10} {'TF-IDF':<10}")
            print("-" * 30)
            for rank, (word, score) in enumerate(word_scores[:10], 1):
                print(f"{rank:<4} {word:<10} {score:<10.4f}")
            
            if len(word_scores) > 10:
                print(f"\n  ... (還有 {len(word_scores) - 10} 個詞)")
            
            if doc_idx == 1:
                print("\n註: 文檔 2-5 的 TF, IDF 計算方式相同,不再贅述。")

    # 建立完整的 TF-IDF DataFrame
    print("\n" + "="*50)
    print("步驟 4: TF-IDF 矩陣 (完整)")
    print("="*50)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=[f"文檔{i+1}" for i in range(len(documents))]
    )

    tfidf_df_nonzero = tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)]
    print(f"\n完整 TF-IDF 矩陣 (顯示前 10 個詞彙):")
    print(tfidf_df_nonzero.iloc[:, :10].round(4))
    if tfidf_df_nonzero.shape[1] > 10:
        print(f"... (還有 {tfidf_df_nonzero.shape[1] - 10} 個詞)")

    # 計算相似度
    print("\n" + "="*50)
    print("步驟 5: 文檔相似度計算 (Cosine Similarity)")
    print("="*50)
    similarity = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        columns=[f"文檔{i+1}" for i in range(len(documents))],
        index=[f"文檔{i+1}" for i in range(len(documents))]
    )

    print("\n相似度矩陣:")
    print(similarity_df.round(4))

    print("\n" + "-"*50)
    print("文檔間相似度分析:")
    print("-"*50)

    similar_pairs = []
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            similar_pairs.append((i+1, j+1, similarity[i][j]))

    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n最相似的文檔對 (Top 5):")
    for rank, (doc1, doc2, score) in enumerate(similar_pairs[:5], 1):
        print(f"\n{rank}. 文檔{doc1} ↔ 文檔{doc2} : 相似度 = {score:.4f}")
        print(f"   文檔{doc1}: {documents[doc1-1]}")
        print(f"   文檔{doc2}: {documents[doc2-1]}")

    print("\n最不相似的文檔對 (Top 3):")
    for rank, (doc1, doc2, score) in enumerate(similar_pairs[-3:], 1):
        print(f"\n{rank}. 文檔{doc1} ↔ 文檔{doc2} : 相似度 = {score:.4f}")
        print(f"   文檔{doc1}: {documents[doc1-1]}")
        print(f"   文檔{doc2}: {documents[doc2-1]}")

    # 儲存結果
    os.makedirs('results', exist_ok=True)
    tfidf_df.to_csv('results/a1_tfidf_matrix.csv', encoding='utf-8-sig')
    similarity_df.to_csv('results/a1_similarity_matrix.csv', encoding='utf-8-sig')
    
    with open('results/a1_top_keywords.txt', 'w', encoding='utf-8') as f:
        for doc_idx in range(len(documents)):
            f.write(f"文檔 {doc_idx + 1}: {documents[doc_idx]}\n")
            f.write("-" * 50 + "\n")
            tfidf_vector = tfidf_matrix[doc_idx].toarray()[0]
            word_scores = [(feature_names[i], tfidf_vector[i]) 
                          for i in range(len(feature_names)) 
                          if tfidf_vector[i] > 0]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (word, score) in enumerate(word_scores[:10], 1):
                f.write(f"{rank:2d}. {word:8s} : {score:.4f}\n")
            f.write("\n")

    print("\n✓ A-1 完成！")


# ============================================================
# A-2: 基於規則的文本分類
# ============================================================
def run_a2():
    """A-2: 基於規則的文本分類"""
    print("\n\nA-2: 基於規則的文本分類 (15 分)")
    print("="*60)

    # 1. 情感分類器
    print("\n1. 情感分類器 (8 分)")
    print("-"*60)

    class RuleBasedSentimentClassifier:
        def __init__(self):
            # 建立正負面詞彙庫
            self.positive_words = ['好', '棒', '優秀', '喜歡', '推薦',
                                  '滿意', '開心', '值得', '精彩', '完美']
            self.negative_words = ['差', '糟', '失望', '討厭', '不推薦',
                                  '浪費', '無聊', '爛', '糟糕', '差勁']
            
            # 加入否定詞處理
            self.negation_words = ['不', '沒', '無', '非', '別']
        
        def classify(self, text):
            words = list(jieba.cut(text))
            positive_count = 0
            negative_count = 0
            
            for i, word in enumerate(words):
                has_negation = False
                if i > 0 and words[i-1] in self.negation_words:
                    has_negation = True
                
                if word in self.positive_words:
                    if has_negation:
                        negative_count += 1
                    else:
                        positive_count += 1
                elif word in self.negative_words:
                    if has_negation:
                        positive_count += 1
                    else:
                        negative_count += 1
            
            if positive_count > negative_count:
                sentiment = "正面"
                confidence = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
            elif negative_count > positive_count:
                sentiment = "負面"
                confidence = negative_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
            else:
                sentiment = "中性"
                confidence = 0.5
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'words': words
            }

    # 2. 主題分類器
    print("\n2. 主題分類器 (7 分)")
    print("-"*60)

    class TopicClassifier:
        def __init__(self):
            self.topic_keywords = {
                '科技': ['AI', '人工智慧', '電腦', '軟體', '程式', '演算法'],
                '運動': ['運動', '健身', '跑步', '游泳', '球類', '比賽'],
                '美食': ['吃', '食物', '餐廳', '美味', '料理', '烹飪'],
                '旅遊': ['旅行', '景點', '飯店', '機票', '觀光', '度假']
            }
        
        def classify(self, text):
            words = set(jieba.cut(text))
            topic_scores = {}
            for topic, keywords in self.topic_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in words)
                topic_scores[topic] = matches
            
            if max(topic_scores.values()) > 0:
                best_topic = max(topic_scores, key=topic_scores.get)
                return {
                    'topic': best_topic,
                    'scores': topic_scores,
                    'confidence': topic_scores[best_topic] / sum(topic_scores.values())
                }
            else:
                return {
                    'topic': '其他',
                    'scores': topic_scores,
                    'confidence': 0
                }

    # 測試資料
    test_texts = [
        "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
        "最新的AI技術突破讓人驚艷，深度學習模型的表現越來越好",
        "這部電影劇情空洞，演技糟糕，完全是浪費時間",
        "每天慢跑5公里，配合適當的重訓，體能進步很多"
    ]

    print("\n測試文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")

    # 測試情感分類器
    print("\n" + "="*60)
    print("情感分類結果:")
    print("="*60)

    sentiment_classifier = RuleBasedSentimentClassifier()
    for i, text in enumerate(test_texts, 1):
        result = sentiment_classifier.classify(text)
        print(f"\n文本 {i}: {text}")
        print(f"  情感: {result['sentiment']}")
        print(f"  信心度: {result['confidence']:.2%}")
        print(f"  正面詞數: {result['positive_count']}, 負面詞數: {result['negative_count']}")
        print(f"  分詞結果: {' / '.join(result['words'][:15])}{'...' if len(result['words']) > 15 else ''}")

    # 測試主題分類器
    print("\n" + "="*60)
    print("主題分類結果:")
    print("="*60)

    topic_classifier = TopicClassifier()
    for i, text in enumerate(test_texts, 1):
        result = topic_classifier.classify(text)
        print(f"\n文本 {i}: {text}")
        print(f"  主題: {result['topic']}")
        print(f"  信心度: {result['confidence']:.2%}")
        print(f"  各主題分數: {result['scores']}")

    # 儲存結果
    os.makedirs('results', exist_ok=True)
    with open('results/a2_classification_results.txt', 'w', encoding='utf-8') as f:
        f.write("A-2: 基於規則的文本分類結果\n")
        f.write("="*60 + "\n\n")
        f.write("情感分類結果:\n")
        f.write("-"*60 + "\n")
        for i, text in enumerate(test_texts, 1):
            result = sentiment_classifier.classify(text)
            f.write(f"\n文本 {i}: {text}\n")
            f.write(f"  情感: {result['sentiment']}\n")
            f.write(f"  信心度: {result['confidence']:.2%}\n")
            f.write(f"  正面詞數: {result['positive_count']}, 負面詞數: {result['negative_count']}\n\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("主題分類結果:\n")
        f.write("-"*60 + "\n")
        for i, text in enumerate(test_texts, 1):
            result = topic_classifier.classify(text)
            f.write(f"\n文本 {i}: {text}\n")
            f.write(f"  主題: {result['topic']}\n")
            f.write(f"  信心度: {result['confidence']:.2%}\n")
            f.write(f"  各主題分數: {result['scores']}\n\n")

    print("\n✓ A-2 完成！")


# ============================================================
# A-3: 統計式自動摘要
# ============================================================
def run_a3():
    """A-3: 統計式自動摘要"""
    print("\n\nA-3: 統計式自動摘要 (15 分)")
    print("="*60)

    class StatisticalSummarizer:
        def __init__(self):
            # 載入停用詞
            self.stop_words = set(['的', '了', '在', '是', '我', '有', '和',
                                  '就', '不', '人', '都', '一', '一個', '上',
                                  '也', '很', '到', '說', '要', '去', '你'])
        
        def sentence_score(self, sentence, word_freq):
            """
            計算句子重要性分數
            考慮因素:
            1. 包含高頻詞的數量
            2. 句子位置 (首尾句加權)
            3. 句子長度 (太短或太長扣分)
            4. 是否包含數字或專有名詞
            """
            # 分詞並過濾停用詞
            words = [w for w in jieba.cut(sentence) if w not in self.stop_words and len(w) > 1]
            
            if len(words) == 0:
                return 0
            
            # 1. 計算高頻詞分數
            word_score = sum(word_freq.get(word, 0) for word in words) / len(words)
            
            # 2. 句子長度分數 (偏好中等長度句子)
            length = len(sentence)
            if length < 10:
                length_score = 0.5
            elif length > 100:
                length_score = 0.7
            else:
                length_score = 1.0
            
            # 3. 檢查是否包含數字
            has_number = any(char.isdigit() for char in sentence)
            number_score = 1.2 if has_number else 1.0
            
            # 綜合分數
            final_score = word_score * length_score * number_score
            
            return final_score
        
        def summarize(self, text, ratio=0.3):
            """
            生成摘要步驟:
            1. 分句 (處理中文標點)
            2. 分詞並計算詞頻
            3. 計算每個句子的重要性分數
            4. 選擇最高分的句子
            5. 按原文順序排列
            """
            # 1. 分句 (處理中文標點)
            import re
            sentences = re.split(r'[。！？\n]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            print(f"\n原文共 {len(sentences)} 個句子")
            print("-" * 60)
            for i, sent in enumerate(sentences, 1):
                print(f"{i}. {sent}")
            
            # 2. 分詞並計算詞頻
            print("\n" + "="*60)
            print("步驟 1: 分詞並計算詞頻")
            print("="*60)
            
            all_words = []
            for sentence in sentences:
                words = [w for w in jieba.cut(sentence) 
                        if w not in self.stop_words and len(w) > 1]
                all_words.extend(words)
            
            # 計算詞頻
            word_freq = {}
            for word in all_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 顯示 Top 15 高頻詞
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 15 高頻詞:")
            for i, (word, freq) in enumerate(sorted_words[:15], 1):
                print(f"  {i:2d}. {word:8s} : {freq} 次")
            
            # 3. 計算每個句子的重要性分數
            print("\n" + "="*60)
            print("步驟 2: 計算句子重要性分數")
            print("="*60)
            
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = self.sentence_score(sentence, word_freq)
                sentence_scores.append((i, sentence, score))
                print(f"\n句子 {i+1} (分數: {score:.4f})")
                print(f"  內容: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            
            # 4. 選擇最高分的句子
            print("\n" + "="*60)
            print("步驟 3: 選擇重要句子")
            print("="*60)
            
            # 根據比例決定摘要句子數量
            summary_length = max(1, int(len(sentences) * ratio))
            print(f"\n摘要比例: {ratio:.1%}")
            print(f"選擇句子數: {summary_length}/{len(sentences)}")
            
            # 按分數排序並選擇 top N
            sentence_scores.sort(key=lambda x: x[2], reverse=True)
            selected_sentences = sentence_scores[:summary_length]
            
            print(f"\n選中的句子:")
            for i, (idx, sent, score) in enumerate(selected_sentences, 1):
                print(f"{i}. 句子 {idx+1} (分數: {score:.4f})")
                print(f"   {sent}")
            
            # 5. 按原文順序排列
            print("\n" + "="*60)
            print("步驟 4: 按原文順序重組摘要")
            print("="*60)
            
            selected_sentences.sort(key=lambda x: x[0])
            summary = ''.join([sent for _, sent, _ in selected_sentences])
            
            print("\n生成的摘要:")
            print("-" * 60)
            print(summary)
            print("-" * 60)
            
            # 統計資訊
            original_length = len(text)
            summary_length_chars = len(summary)
            compression_ratio = (1 - summary_length_chars / original_length) * 100
            
            print(f"\n壓縮統計:")
            print(f"  原文字數: {original_length} 字")
            print(f"  摘要字數: {summary_length_chars} 字")
            print(f"  壓縮率: {compression_ratio:.1f}%")
            
            return {
                'summary': summary,
                'selected_sentences': selected_sentences,
                'original_length': original_length,
                'summary_length': summary_length_chars,
                'compression_ratio': compression_ratio
            }
    
    # 測試文章
    article = """
人工智慧（AI）的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。

在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。

教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。

然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。

面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
"""
    
    print("\n測試文章:")
    print("="*60)
    print(article.strip())
    print("="*60)
    
    # 測試摘要系統
    print("\n\n開始生成摘要...")
    print("="*60)
    
    summarizer = StatisticalSummarizer()
    
    # 測試不同的摘要比例
    for ratio in [0.3, 0.5]:
        print("\n\n" + "="*60)
        print(f"摘要比例: {ratio:.0%}")
        print("="*60)
        result = summarizer.summarize(article, ratio=ratio)
    
    # 儲存結果
    print("\n\n" + "="*60)
    print("儲存結果")
    print("="*60)
    
    os.makedirs('results', exist_ok=True)
    
    all_results = {}
    with open('results/a3_summary_results.txt', 'w', encoding='utf-8') as f:
        f.write("A-3: 統計式自動摘要結果\n")
        f.write("="*60 + "\n\n")
        
        f.write("原文:\n")
        f.write("-"*60 + "\n")
        f.write(article.strip() + "\n\n")
        
        for ratio in [0.3, 0.5]:
            f.write("\n" + "="*60 + "\n")
            f.write(f"摘要比例: {ratio:.0%}\n")
            f.write("="*60 + "\n\n")
            
            result = summarizer.summarize(article, ratio=ratio)
            all_results[ratio] = result
            
            f.write("摘要:\n")
            f.write("-"*60 + "\n")
            f.write(result['summary'] + "\n\n")
            
            f.write(f"原文字數: {result['original_length']} 字\n")
            f.write(f"摘要字數: {result['summary_length']} 字\n")
            f.write(f"壓縮率: {result['compression_ratio']:.1f}%\n\n")
    
    print("✓ 摘要結果已儲存至: results/a3_summary_results.txt")
    
    # ========== 新增：視覺化 ==========
    print("\n" + "="*60)
    print("生成視覺化圖表")
    print("="*60)
    
    # 1. 摘要壓縮率比較圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ratios = [0.3, 0.5]
    compression_ratios = [all_results[r]['compression_ratio'] for r in ratios]
    summary_lengths = [all_results[r]['summary_length'] for r in ratios]
    original_length = all_results[0.3]['original_length']
    
    # 左圖：壓縮率比較
    bars = axes[0].bar([f'{int(r*100)}%' for r in ratios], compression_ratios, 
                       color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_ylabel('壓縮率 (%)', fontsize=12)
    axes[0].set_title('不同摘要比例的壓縮率', fontsize=14, pad=20)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # 右圖：字數對比
    x = np.arange(len(ratios))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, [original_length]*len(ratios), width, 
                        label='原文', color='#95E1D3')
    bars2 = axes[1].bar(x + width/2, summary_lengths, width,
                        label='摘要', color='#F38181')
    
    axes[1].set_ylabel('字數', fontsize=12)
    axes[1].set_title('原文與摘要字數對比', fontsize=14, pad=20)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{int(r*100)}%' for r in ratios])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 在長條上顯示數值
    for bar in bars1:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/a3_summary_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 摘要比較圖已儲存至: results/a3_summary_comparison.png")
    plt.close()
    
    # 2. 句子重要性分數分布
    # 重新計算一次以獲取句子分數
    import re
    sentences = re.split(r'[。！？\n]+', article)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    all_words = []
    for sentence in sentences:
        words = [w for w in jieba.cut(sentence) 
                if w not in summarizer.stop_words and len(w) > 1]
        all_words.extend(words)
    
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sentence_scores = []
    for sentence in sentences:
        score = summarizer.sentence_score(sentence, word_freq)
        sentence_scores.append(score)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(sentence_scores)+1), sentence_scores, 
             marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('句子編號', fontsize=12)
    plt.ylabel('重要性分數', fontsize=12)
    plt.title('各句子重要性分數分布', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    # 標記最重要的句子
    top_indices = np.argsort(sentence_scores)[-3:][::-1]
    for idx in top_indices:
        plt.scatter(idx+1, sentence_scores[idx], color='red', s=200, zorder=5, alpha=0.6)
        plt.annotate(f'Top {list(top_indices).index(idx)+1}', 
                    xy=(idx+1, sentence_scores[idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/a3_sentence_scores.png', dpi=300, bbox_inches='tight')
    print("✓ 句子分數分布圖已儲存至: results/a3_sentence_scores.png")
    plt.close()
    
    # 3. 詞雲視覺化 - 顯示高頻關鍵詞 (額外加分項目)
    print("\n生成關鍵詞詞雲...")
    
    # 取得所有詞的頻率
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    word_freq_dict = dict(sorted_words[:50])  # 取 Top 50
    
    if word_freq_dict:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左圖：原文關鍵詞詞雲
        wordcloud1 = WordCloud(
            font_path='C:/Windows/Fonts/msjh.ttc',
            width=800, 
            height=600,
            background_color='white',
            colormap='Blues',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq_dict)
        
        axes[0].imshow(wordcloud1, interpolation='bilinear')
        axes[0].set_title('原文關鍵詞詞雲 (依詞頻)', fontsize=14, pad=10)
        axes[0].axis('off')
        
        # 右圖：使用不同配色的詞雲
        wordcloud2 = WordCloud(
            font_path='C:/Windows/Fonts/msjh.ttc',
            width=800, 
            height=600,
            background_color='#1a1a1a',
            colormap='plasma',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq_dict)
        
        axes[1].imshow(wordcloud2, interpolation='bilinear')
        axes[1].set_title('原文關鍵詞詞雲 (暗色主題)', fontsize=14, pad=10)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/a3_wordcloud.png', dpi=300, bbox_inches='tight')
        print("✓ 關鍵詞詞雲已儲存至: results/a3_wordcloud.png")
        plt.close()
    
    print("\n" + "="*60)
    print("✓ A-3 完成！")
    print("="*60)


# ============================================================
# 主程式
# ============================================================
def main(task=None):
    """
    主程式入口
    task: 'A1', 'A2', 'A3', 'ALL', None
    """
    print("Part A: 傳統方法")
    print("="*60)
    
    if task == 'A1':
        run_a1()
    elif task == 'A2':
        run_a2()
    elif task == 'A3':
        run_a3()
    elif task == 'ALL' or task is None:
        run_a1()
        run_a2()
        run_a3()
    else:
        print(f"⚠ 無效的任務: {task}")
        return
    
    print("\n" + "="*60)
    print("🎉 Part A 完成！")
    print("="*60)
    print("\n結果檔案位於 results/ 資料夾")


if __name__ == "__main__":
    # 支援命令列參數
    if len(sys.argv) > 1:
        task = sys.argv[1].upper()
        main(task)
    else:
        main('ALL')