# 作業2 - 從傳統到現代文本處理方法實作與比較

## 安裝與執行
1. 安裝相依套件：`pip install -r requirements.txt`
2. 設定 OpenAI API Key：複製 `.env.example` 為 `.env` 並填入 key
3. 執行程式：`python run_all.py`

## 專案結構
- `traditional_methods.py` - 傳統 TF-IDF 方法
- `modern_methods.py` - GPT-4o AI 方法
- `comparison.py` - 比較分析
- `results/` - 執行結果輸出
```

3. **建立 `.env.example`**：
```
OPENAI_API_KEY=your_api_key_here
```

4. **確保 `requirements.txt` 完整**，需包含：
```
jieba
scikit-learn
numpy
matplotlib
wordcloud
openai>=2.0
python-dotenv