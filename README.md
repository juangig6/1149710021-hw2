# 作業2 - 從傳統到現代文本處理方法實作與比較

學號：114971021  
姓名：Yi-Kuei,Chuang

## 📖 專案簡介

本專案實作並比較傳統文本處理方法（TF-IDF）與現代 AI 方法（GPT-4o）在中文文本處理任務上的表現。透過實作相似度計算、文本分類與自動摘要三項任務，深入分析兩種方法的優缺點與適用場景。

## 🚀 主要功能

### Part A - 傳統方法實作
- **A-1 TF-IDF 文本相似度計算**：計算五份中文文檔的相似度矩陣
- **A-2 基於規則的文本分類**：實作情感分析與主題分類
- **A-3 統計式自動摘要**：使用詞頻統計提取重要句子

### Part B - 現代 AI 方法
- **B-1 語意相似度計算**：使用 GPT-4o 判斷語意相似度
- **B-2 AI 文本分類**：多維度智能分類（情感、主題）
- **B-3 AI 自動摘要**：生成式摘要，自動產生流暢摘要

### Part C - 比較分析
- **C-1 量化比較**：準確率、處理時間、成本等指標比較
- **C-2 質性分析**：方法特性、限制與適用場景分析

### 加分題
- **Word2Vec 實作**：額外實作 Word2Vec 與 TF-IDF 的比較

## 💻 環境需求

- Python 3.8 或以上版本
- OpenAI API Key（用於 Part B）

## 📦 安裝步驟

1. **Clone 專案**
```bash
git clone https://github.com/juangig6/1149710021-hw2.git
cd 1149710021-hw2
```

2. **安裝相依套件**
```bash
pip install -r requirements.txt
```

3. **設定 API Key**
```bash
# 複製環境變數範例檔
cp .env.example .env

# 編輯 .env 檔案，加入您的 OpenAI API Key
# OPENAI_API_KEY=your_api_key_here
```

## 🎯 執行方式

### 執行完整作業
```bash
python run_all.py
```

### 執行個別部分
程式提供互動式選單，可選擇執行：
- 選項 1：執行完整作業 (Part A + B + C)
- 選項 2：只執行 Part A（傳統方法）
- 選項 3：只執行 Part B（AI 方法）
- 選項 4：只執行 Part C（比較分析）

## 📁 專案結構
```
1149710021-hw2/
│
├── run_all.py                 # 主程式入口
├── traditional_methods.py     # Part A: 傳統方法實作
├── modern_methods.py          # Part B: AI 方法實作
├── comparison.py              # Part C: 比較分析
├── word2vec_final.py          # 加分題：Word2Vec 實作
│
├── requirements.txt           # Python 套件清單
├── .env.example              # 環境變數範例
├── README.md                 # 專案說明文件
│
└── results/                  # 執行結果輸出資料夾
    ├── a1_similarity_matrix.csv
    ├── a2_classification_results.json
    ├── a3_summary_results.txt
    ├── b1_similarity_results.json
    ├── b2_classification_results.json
    ├── b3_summary_results.json
    ├── c1_quantitative_comparison.json
    └── c2_qualitative_analysis.md
```

## 📊 執行結果

執行程式後，所有結果會自動儲存在 `results/` 資料夾中：

- **相似度計算結果**：CSV 矩陣與 JSON 格式
- **分類結果**：包含信心度的 JSON 檔案
- **摘要結果**：原文、摘要與壓縮率統計
- **比較分析**：量化數據與質性分析報告
- **視覺化圖表**：詞雲、分數分布圖等

## 🔑 主要發現

### 傳統方法（TF-IDF）
- ✅ 速度快（< 0.1 秒）
- ✅ 完全離線、保護隱私
- ✅ 結果可解釋性高
- ❌ 無法理解深層語意
- ❌ 處理複雜語言現象能力有限

### 現代 AI 方法（GPT-4o）
- ✅ 語意理解能力強（準確率 85-92%）
- ✅ 生成品質優秀
- ✅ 任務定義彈性高
- ❌ 需要 API 成本
- ❌ 處理速度較慢（1-3 秒）

## 📝 報告文件

詳細的實作過程與分析請參考：
- [作業2-從傳統到現代文本處理方法實作與比較報告.pdf](./作業2-從傳統到現代文本處理方法實作與比較報告.pdf)

## 🛠️ 技術堆疊

- **自然語言處理**：jieba, scikit-learn
- **深度學習**：gensim (Word2Vec)
- **AI API**：OpenAI GPT-4o
- **視覺化**：matplotlib, wordcloud
- **資料處理**：numpy, pandas

## 📌 注意事項

1. **API Key 安全**：請勿將真實的 API Key 上傳到 GitHub
2. **執行時間**：Part B 需要網路連接，可能需要等待 API 回應
3. **成本考量**：使用 OpenAI API 會產生費用，請注意使用量

## 🤝 作者

- **姓名**：莊一桂
- **學號**：114971021
- **課程**：自然語言處理

## 📅 更新日誌

- 2024-11-23：初始版本上傳
- 2024-11-23：新增 Word2Vec 加分題實作
- 2024-11-23：完成所有測試與文件

## 📜 授權

本專案為課程作業，僅供教學使用。

---

如有任何問題或建議，歡迎提出 Issue 或 Pull Request！