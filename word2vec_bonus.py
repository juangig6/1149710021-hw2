"""
Word2Vec 文本處理方法 - 加分題實作
比較 Word2Vec 與 TF-IDF 在文本相似度計算上的差異
"""

import numpy as np
import pandas as pd
import json
import jieba
import warnings
warnings.filterwarnings('ignore')

# Gensim for Word2Vec
try:
    from gensim.models import Word2Vec
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("警告: gensim 未安裝，將使用模擬數據")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Word2VecAnalyzer:
    """Word2Vec 文本分析器"""
    
    def __init__(self):
        self.model = None
        self.word_vectors = {}
        self.doc_vectors = {}
        
    def train_word2vec(self, texts, vector_size=100, window=5, min_count=1):
        """訓練 Word2Vec 模型"""
        print("\n訓練 Word2Vec 模型...")
        
        # 分詞
        tokenized_texts = []
        for text in texts:
            words = list(jieba.cut(text))
            # 過濾標點符號
            words = [w for w in words if w.strip() and w not in '，。！？、']
            tokenized_texts.append(words)
        
        if GENSIM_AVAILABLE:
            # 訓練模型
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4,
                epochs=100,
                seed=42
            )
            print(f"  模型訓練完成: {len(self.model.wv)} 個詞彙")
            return tokenized_texts
        else:
            print("  使用隨機向量模擬 Word2Vec")
            # 建立詞彙表
            vocab = set()
            for words in tokenized_texts:
                vocab.update(words)
            
            # 為每個詞生成隨機向量
            np.random.seed(42)
            for word in vocab:
                self.word_vectors[word] = np.random.randn(vector_size)
            
            return tokenized_texts
    
    def get_word_vector(self, word):
        """獲取詞向量"""
        if GENSIM_AVAILABLE and self.model:
            try:
                return self.model.wv[word]
            except KeyError:
                return np.zeros(self.model.wv.vector_size)
        else:
            return self.word_vectors.get(word, np.zeros(100))
    
    def compute_doc_vector(self, words):
        """計算文檔向量 (詞向量平均)"""
        vectors = []
        for word in words:
            vec = self.get_word_vector(word)
            if vec is not None and np.any(vec):
                vectors.append(vec)
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(100 if not self.model else self.model.wv.vector_size)
    
    def compute_similarity_matrix(self, tokenized_texts):
        """計算文檔相似度矩陣"""
        print("\n計算 Word2Vec 相似度矩陣...")
        
        # 計算每個文檔的向量
        doc_vectors = []
        for words in tokenized_texts:
            vec = self.compute_doc_vector(words)
            doc_vectors.append(vec)
        
        # 計算相似度矩陣
        similarity_matrix = cosine_similarity(doc_vectors)
        return similarity_matrix
    
    def find_similar_words(self, word, topn=5):
        """找出相似詞彙"""
        if GENSIM_AVAILABLE and self.model:
            try:
                similar = self.model.wv.most_similar(word, topn=topn)
                return similar
            except KeyError:
                return []
        else:
            # 模擬相似詞
            return [("相似詞1", 0.8), ("相似詞2", 0.7)]

def compare_methods(texts):
    """比較 TF-IDF 和 Word2Vec 方法"""
    
    print("=" * 60)
    print("Word2Vec vs TF-IDF 比較分析")
    print("=" * 60)
    
    # 1. TF-IDF 方法
    print("\n【方法 1: TF-IDF】")
    print("-" * 40)
    
    start_time = time.time()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_similarity = cosine_similarity(tfidf_matrix)
    tfidf_time = time.time() - start_time
    
    print(f"執行時間: {tfidf_time:.4f} 秒")
    print(f"特徵維度: {tfidf_matrix.shape[1]}")
    print(f"詞彙量: {len(vectorizer.vocabulary_)}")
    
    # 2. Word2Vec 方法
    print("\n【方法 2: Word2Vec】")
    print("-" * 40)
    
    analyzer = Word2VecAnalyzer()
    start_time = time.time()
    tokenized = analyzer.train_word2vec(texts)
    w2v_similarity = analyzer.compute_similarity_matrix(tokenized)
    w2v_time = time.time() - start_time
    
    print(f"執行時間: {w2v_time:.4f} 秒")
    print(f"向量維度: 100")
    
    # 3. 相似度比較
    print("\n【相似度矩陣比較】")
    print("-" * 40)
    
    doc_names = [f"文檔{i+1}" for i in range(len(texts))]
    
    print("\nTF-IDF 相似度矩陣:")
    tfidf_df = pd.DataFrame(tfidf_similarity, index=doc_names, columns=doc_names)
    print(tfidf_df.round(4))
    
    print("\nWord2Vec 相似度矩陣:")
    w2v_df = pd.DataFrame(w2v_similarity, index=doc_names, columns=doc_names)
    print(w2v_df.round(4))
    
    # 4. 差異分析
    print("\n【差異分析】")
    print("-" * 40)
    
    diff_matrix = np.abs(tfidf_similarity - w2v_similarity)
    avg_diff = np.mean(diff_matrix[np.triu_indices_from(diff_matrix, k=1)])
    
    print(f"平均差異: {avg_diff:.4f}")
    print(f"最大差異: {np.max(diff_matrix):.4f}")
    
    # 找出差異最大的文檔對
    max_diff_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
    if max_diff_idx[0] != max_diff_idx[1]:
        print(f"差異最大的文檔對: {doc_names[max_diff_idx[0]]} vs {doc_names[max_diff_idx[1]]}")
        print(f"  TF-IDF 相似度: {tfidf_similarity[max_diff_idx]:.4f}")
        print(f"  Word2Vec 相似度: {w2v_similarity[max_diff_idx]:.4f}")
    
    # 5. 詞彙相似度展示 (Word2Vec 特有功能)
    if GENSIM_AVAILABLE and analyzer.model:
        print("\n【Word2Vec 詞彙相似度示例】")
        print("-" * 40)
        
        test_words = ['人工智慧', '學習', '運動']
        for word in test_words:
            print(f"\n與 '{word}' 相似的詞:")
            similar = analyzer.find_similar_words(word)
            for sim_word, score in similar[:3]:
                print(f"  - {sim_word}: {score:.4f}")
    
    # 6. 視覺化
    create_comparison_visualization(tfidf_similarity, w2v_similarity, texts)
    
    # 7. 返回結果
    return {
        'tfidf': {
            'similarity_matrix': tfidf_similarity.tolist(),
            'execution_time': tfidf_time,
            'dimensions': int(tfidf_matrix.shape[1])
        },
        'word2vec': {
            'similarity_matrix': w2v_similarity.tolist(),
            'execution_time': w2v_time,
            'dimensions': 100
        },
        'comparison': {
            'average_difference': float(avg_diff),
            'max_difference': float(np.max(diff_matrix))
        }
    }

def create_comparison_visualization(tfidf_sim, w2v_sim, texts):
    """創建比較視覺化圖表"""
    
    print("\n生成視覺化圖表...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    doc_labels = [f"D{i+1}" for i in range(len(texts))]
    
    # TF-IDF 熱圖
    sns.heatmap(tfidf_sim, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=doc_labels, yticklabels=doc_labels,
                vmin=0, vmax=1, square=True, ax=axes[0],
                cbar_kws={'label': '相似度'})
    axes[0].set_title('TF-IDF 相似度矩陣', fontsize=14, fontweight='bold')
    
    # Word2Vec 熱圖
    sns.heatmap(w2v_sim, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=doc_labels, yticklabels=doc_labels,
                vmin=0, vmax=1, square=True, ax=axes[1],
                cbar_kws={'label': '相似度'})
    axes[1].set_title('Word2Vec 相似度矩陣', fontsize=14, fontweight='bold')
    
    # 差異熱圖
    diff_matrix = np.abs(tfidf_sim - w2v_sim)
    sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='Purples',
                xticklabels=doc_labels, yticklabels=doc_labels,
                vmin=0, vmax=0.5, square=True, ax=axes[2],
                cbar_kws={'label': '差異值'})
    axes[2].set_title('方法差異矩陣 (|TF-IDF - Word2Vec|)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 儲存圖片
    output_path = Path('results/word2vec_comparison.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 視覺化已儲存至: {output_path}")

def generate_analysis_report(results):
    """生成分析報告"""
    
    report = f"""
================================================================================
Word2Vec 加分題 - 方法比較分析報告
================================================================================

一、技術原理比較
--------------------------------------------------------------------------------

1. TF-IDF (Term Frequency-Inverse Document Frequency)
   - 原理：基於詞頻統計的向量空間模型
   - 特點：稀疏向量、高維度、可解釋性強
   - 優勢：簡單高效、無需訓練、適合關鍵詞提取
   - 劣勢：忽略詞序、無法捕捉語義關係

2. Word2Vec (Word to Vector)
   - 原理：基於神經網絡的詞嵌入模型
   - 特點：密集向量、低維度、語義表示
   - 優勢：捕捉語義關係、支援詞彙相似度計算
   - 劣勢：需要訓練、依賴訓練數據質量

二、實驗結果分析
--------------------------------------------------------------------------------

執行效能比較：
- TF-IDF 執行時間：{results['tfidf']['execution_time']:.4f} 秒
- Word2Vec 執行時間：{results['word2vec']['execution_time']:.4f} 秒
- 速度差異：TF-IDF 快 {results['word2vec']['execution_time']/results['tfidf']['execution_time']:.2f} 倍

向量維度比較：
- TF-IDF 維度：{results['tfidf']['dimensions']} 維（詞彙量決定）
- Word2Vec 維度：{results['word2vec']['dimensions']} 維（預設固定）
- 維度差異：TF-IDF 高 {results['tfidf']['dimensions']/results['word2vec']['dimensions']:.1f} 倍

相似度差異：
- 平均差異：{results['comparison']['average_difference']:.4f}
- 最大差異：{results['comparison']['max_difference']:.4f}

三、深入洞察
--------------------------------------------------------------------------------

1. 語義理解能力差異
   - TF-IDF：僅考慮詞彙重疊，"人工智慧"和"AI"視為不同詞
   - Word2Vec：學習語義關係，能識別同義詞和相關概念

2. 適用場景分析
   - TF-IDF 適合：文檔檢索、關鍵詞提取、主題分類
   - Word2Vec 適合：語義相似度、詞彙類比、情感分析

3. 混合策略建議
   - 第一階段：使用 TF-IDF 快速篩選候選文檔
   - 第二階段：使用 Word2Vec 精確排序和語義匹配
   - 優勢互補：結合統計特徵和語義特徵

四、技術實作要點
--------------------------------------------------------------------------------

1. Word2Vec 參數調優
   - vector_size：向量維度，建議 100-300
   - window：上下文窗口，建議 5-10
   - min_count：最小詞頻，過濾低頻詞
   - epochs：訓練輪數，建議 50-100

2. 中文處理注意事項
   - 分詞品質直接影響效果
   - 可考慮加入自定義詞典
   - 停用詞處理需謹慎

3. 評估指標建議
   - 內部評估：詞彙相似度合理性
   - 外部評估：下游任務表現
   - 人工評估：抽樣檢查結果

五、結論與建議
--------------------------------------------------------------------------------

Word2Vec 作為深度學習時代的代表性技術，在語義理解上確實優於傳統的 TF-IDF，
但並非在所有場景都是最佳選擇。實際應用中應根據具體需求選擇：

- 需要快速處理、結果可解釋 → TF-IDF
- 需要語義理解、詞彙關係 → Word2Vec
- 追求最佳效果 → 混合使用或升級到 BERT 等預訓練模型

未來展望：
隨著 Transformer 架構的發展，BERT、GPT 等預訓練語言模型提供了更強大的
文本表示能力，但 Word2Vec 因其簡單高效，在特定場景仍有其價值。

================================================================================
報告生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""
    
    return report

def main():
    """主程式"""
    
    print("=" * 60)
    print("Word2Vec 加分題實作")
    print("=" * 60)
    
    # 測試文檔
    texts = [
        "人工智慧正在改變世界，機器學習是其核心技術",
        "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
        "今天天氣很好，適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康，每天都應該保持運動習慣"
    ]
    
    print("\n測試文檔：")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    # 執行比較分析
    results = compare_methods(texts)
    
    # 生成報告
    report = generate_analysis_report(results)
    print(report)
    
    # 儲存結果
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # 儲存 JSON 結果
    with open(output_dir / 'word2vec_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON 結果已儲存至: results/word2vec_comparison.json")
    
    # 儲存文字報告
    with open(output_dir / 'word2vec_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ 分析報告已儲存至: results/word2vec_analysis_report.txt")
    
    print("\n" + "=" * 60)
    print("✅ Word2Vec 加分題完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
