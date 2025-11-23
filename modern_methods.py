#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part B: 現代 AI 方法 (30 分)
使用 OpenAI gpt-4o API 完成相同任務
包含 B-1, B-2, B-3 三個任務
"""
import sys
import os
import json
from openai import OpenAI

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()  # 從 .env 檔案載入環境變數
    print("✓ 已載入 .env 檔案")
except ImportError:
    print("ℹ python-dotenv 未安裝，使用系統環境變數")
except Exception as e:
    print(f"ℹ 載入 .env 檔案時發生錯誤: {e}")

# ============================================================
# B-1: 語意相似度計算 (10 分)
# ============================================================
def run_b1():
    """B-1: 語意相似度計算"""
    print("\nB-1: 語意相似度計算 (10 分)")
    print("="*60)
    print("使用 gpt-4o 判斷語意相似度\n")
    
    def ai_similarity(text1, text2, api_key):
        """
        使用 gpt-4o 判斷語意相似度
        
        要求:
        1. 設計適當的 prompt
        2. 返回 0-100 的相似度分數
        3. 處理 API 錯誤
        """
        try:
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
請評估以下兩段文字的語意相似度。
考慮因素:
1. 主題相關性
2. 語意重疊程度
3. 表達的觀點是否一致

文字1: {text1}
文字2: {text2}

請只回答一個0-100的數字，代表相似度百分比。
數字越高表示越相似，不需要其他說明。
"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一個專業的文本相似度分析助手，只需回答數字。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            # 提取相似度分數
            result = response.choices[0].message.content.strip()
            
            # 嘗試從回應中提取數字
            import re
            numbers = re.findall(r'\d+', result)
            if numbers:
                similarity_score = int(numbers[0])
                # 確保在 0-100 範圍內
                similarity_score = max(0, min(100, similarity_score))
            else:
                similarity_score = 50  # 默認值
            
            return {
                'similarity': similarity_score,
                'raw_response': result
            }
            
        except Exception as e:
            print(f"⚠ API 錯誤: {e}")
            return {
                'similarity': -1,
                'error': str(e)
            }
    
    # 測試數據
    test_pairs = [
        ("人工智慧正在改變世界", "AI技術revolutionizing我們的生活"),
        ("今天天氣很好", "股市今天上漲了"),
        ("我喜歡吃披薩", "披薩是我最愛的食物"),
        ("機器學習是AI的一部分", "深度學習屬於機器學習領域")
    ]
    
    print("測試文本對:")
    print("-"*60)
    for i, (text1, text2) in enumerate(test_pairs, 1):
        print(f"\n{i}. 文字1: {text1}")
        print(f"   文字2: {text2}")
    
    # 檢查 API Key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n" + "="*60)
        print("⚠ 警告: 未設置 OPENAI_API_KEY 環境變數")
        print("="*60)
        print("\n請設置環境變數:")
        print("  Windows: set OPENAI_API_KEY=your-api-key")
        print("  Linux/Mac: export OPENAI_API_KEY=your-api-key")
        print("\n或在程式中直接設置:")
        print("  api_key = 'your-api-key'")
        print("\n使用模擬數據進行演示...")
        
        # 使用模擬數據
        print("\n" + "="*60)
        print("相似度計算結果 (模擬)")
        print("="*60)
        
        mock_results = [85, 15, 92, 88]
        for i, ((text1, text2), score) in enumerate(zip(test_pairs, mock_results), 1):
            print(f"\n{i}. 文字1: {text1}")
            print(f"   文字2: {text2}")
            print(f"   相似度: {score}%")
    else:
        # 實際調用 API
        print("\n" + "="*60)
        print("相似度計算結果")
        print("="*60)
        
        results = []
        for i, (text1, text2) in enumerate(test_pairs, 1):
            print(f"\n正在計算第 {i} 對相似度...")
            result = ai_similarity(text1, text2, api_key)
            results.append(result)
            
            print(f"{i}. 文字1: {text1}")
            print(f"   文字2: {text2}")
            if 'error' in result:
                print(f"   錯誤: {result['error']}")
            else:
                print(f"   相似度: {result['similarity']}%")
                print(f"   原始回應: {result['raw_response']}")
        
        # 儲存結果
        os.makedirs('results', exist_ok=True)
        with open('results/b1_similarity_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'test_pairs': [{'text1': t1, 'text2': t2} for t1, t2 in test_pairs],
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print("\n✓ 結果已儲存至: results/b1_similarity_results.json")
    
    print("\n" + "="*60)
    print("✓ B-1 完成！")
    print("="*60)


# ============================================================
# B-2: AI 文本分類 (10 分)
# ============================================================
def run_b2():
    """B-2: AI 文本分類"""
    print("\n\nB-2: AI 文本分類 (10 分)")
    print("="*60)
    print("使用 gpt-4o 進行多維度分類\n")
    
    def ai_classify(text, api_key):
        """
        使用 gpt-4o 進行多維度分類
        
        返回格式:
        {
            "sentiment": "正面/負面/中性",
            "topic": "主題類別",
            "confidence": 0.95
        }
        """
        try:
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
請分析以下文本的情感和主題:

文本: {text}

請以JSON格式回答，包含以下欄位:
1. sentiment: 情感分類 (正面/負面/中性)
2. topic: 主題分類 (科技/運動/美食/旅遊/其他)
3. confidence: 信心度 (0.0-1.0)

只需回答JSON，不要其他說明。
範例格式:
{{"sentiment": "正面", "topic": "美食", "confidence": 0.95}}
"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一個專業的文本分類助手，回答必須是有效的JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            # 清理可能的 markdown 代碼塊標記
            result = result.replace('```json', '').replace('```', '').strip()
            
            # 解析 JSON
            classification = json.loads(result)
            
            return classification
            
        except json.JSONDecodeError as e:
            print(f"⚠ JSON 解析錯誤: {e}")
            print(f"原始回應: {result}")
            return {
                'sentiment': '未知',
                'topic': '其他',
                'confidence': 0.0,
                'error': 'JSON解析失敗'
            }
        except Exception as e:
            print(f"⚠ API 錯誤: {e}")
            return {
                'sentiment': '未知',
                'topic': '其他',
                'confidence': 0.0,
                'error': str(e)
            }
    
    # 測試數據
    test_texts = [
        "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
        "最新的AI技術突破讓人驚艷，深度學習模型的表現越來越好",
        "這部電影劇情空洞，演技糟糕，完全是浪費時間",
        "每天慢跑5公里，配合適當的重訓，體能進步很多"
    ]
    
    print("測試文本:")
    print("-"*60)
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    # 檢查 API Key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n" + "="*60)
        print("⚠ 警告: 未設置 OPENAI_API_KEY 環境變數")
        print("="*60)
        print("\n使用模擬數據進行演示...")
        
        # 使用模擬數據
        print("\n" + "="*60)
        print("分類結果 (模擬)")
        print("="*60)
        
        mock_results = [
            {"sentiment": "正面", "topic": "美食", "confidence": 0.95},
            {"sentiment": "正面", "topic": "科技", "confidence": 0.92},
            {"sentiment": "負面", "topic": "其他", "confidence": 0.88},
            {"sentiment": "正面", "topic": "運動", "confidence": 0.90}
        ]
        
        for i, (text, result) in enumerate(zip(test_texts, mock_results), 1):
            print(f"\n文本 {i}: {text}")
            print(f"  情感: {result['sentiment']}")
            print(f"  主題: {result['topic']}")
            print(f"  信心度: {result['confidence']:.2f}")
    else:
        # 實際調用 API
        print("\n" + "="*60)
        print("分類結果")
        print("="*60)
        
        results = []
        for i, text in enumerate(test_texts, 1):
            print(f"\n正在分類第 {i} 個文本...")
            result = ai_classify(text, api_key)
            results.append(result)
            
            print(f"文本 {i}: {text}")
            print(f"  情感: {result.get('sentiment', '未知')}")
            print(f"  主題: {result.get('topic', '其他')}")
            print(f"  信心度: {result.get('confidence', 0.0):.2f}")
            if 'error' in result:
                print(f"  錯誤: {result['error']}")
        
        # 儲存結果
        os.makedirs('results', exist_ok=True)
        with open('results/b2_classification_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'test_texts': test_texts,
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print("\n✓ 結果已儲存至: results/b2_classification_results.json")
    
    print("\n" + "="*60)
    print("✓ B-2 完成！")
    print("="*60)


# ============================================================
# B-3: AI 自動摘要 (10 分)
# ============================================================
def run_b3():
    """B-3: AI 自動摘要"""
    print("\n\nB-3: AI 自動摘要 (10 分)")
    print("="*60)
    print("使用 gpt-4o 生成摘要\n")
    
    def ai_summarize(text, max_length, api_key):
        """
        使用 gpt-4o 生成摘要
        
        要求:
        1. 可控制摘要長度
        2. 保留關鍵資訊
        3. 語句通順
        """
        try:
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
請為以下文章生成摘要。

要求:
1. 摘要長度不超過 {max_length} 字
2. 保留文章的關鍵資訊和主要論點
3. 語句通順，邏輯清晰
4. 不要添加原文沒有的內容

文章:
{text}

請直接輸出摘要內容，不要其他說明。
"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一個專業的文章摘要助手，擅長提取關鍵資訊並生成簡潔摘要。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=max_length * 2  # 考慮 token 與字數的關係
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                'summary': summary,
                'length': len(summary),
                'compression_ratio': (1 - len(summary) / len(text)) * 100
            }
            
        except Exception as e:
            print(f"⚠ API 錯誤: {e}")
            return {
                'summary': '',
                'length': 0,
                'compression_ratio': 0,
                'error': str(e)
            }
    
    # 測試文章
    article = """
人工智慧（AI）的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。

在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。

教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。

然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。

面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
"""
    
    print("測試文章:")
    print("="*60)
    print(article.strip())
    print("="*60)
    print(f"\n原文字數: {len(article)} 字")
    
    # 檢查 API Key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n" + "="*60)
        print("⚠ 警告: 未設置 OPENAI_API_KEY 環境變數")
        print("="*60)
        print("\n使用模擬數據進行演示...")
        
        # 使用模擬數據
        print("\n" + "="*60)
        print("摘要結果 (模擬)")
        print("="*60)
        
        mock_summary_100 = "人工智慧正在改變我們的生活，在醫療和教育領域帶來革新。然而也面臨就業、隱私和倫理等挑戰，需要建立相應的法規和準則。"
        mock_summary_150 = "人工智慧（AI）深刻改變我們的生活方式，在醫療領域協助診斷、提高準確率，在教育方面提供個人化學習。但AI發展也帶來就業、隱私和倫理挑戰，需要在推動發展的同時建立相應的法律法規和倫理準則，確保AI為人類福祉服務。"
        
        for max_len, summary in [(100, mock_summary_100), (150, mock_summary_150)]:
            print(f"\n摘要長度限制: {max_len} 字")
            print("-"*60)
            print(f"摘要: {summary}")
            print(f"實際長度: {len(summary)} 字")
            print(f"壓縮率: {(1 - len(summary) / len(article)) * 100:.1f}%")
    else:
        # 實際調用 API
        print("\n" + "="*60)
        print("摘要結果")
        print("="*60)
        
        results = {}
        for max_length in [100, 150]:
            print(f"\n正在生成 {max_length} 字摘要...")
            result = ai_summarize(article, max_length, api_key)
            results[max_length] = result
            
            print(f"\n摘要長度限制: {max_length} 字")
            print("-"*60)
            if 'error' in result:
                print(f"錯誤: {result['error']}")
            else:
                print(f"摘要: {result['summary']}")
                print(f"實際長度: {result['length']} 字")
                print(f"壓縮率: {result['compression_ratio']:.1f}%")
        
        # 儲存結果
        os.makedirs('results', exist_ok=True)
        with open('results/b3_summary_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'article': article,
                'article_length': len(article),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print("\n✓ 結果已儲存至: results/b3_summary_results.json")
    
    print("\n" + "="*60)
    print("✓ B-3 完成！")
    print("="*60)


# ============================================================
# 主程式
# ============================================================
def main(task=None):
    """
    主程式入口
    task: 'B1', 'B2', 'B3', 'ALL', None
    """
    print("Part B: 現代 AI 方法 (使用 OpenAI gpt-4o)")
    print("="*60)
    
    # 檢查 OpenAI 套件
    try:
        import openai
        print(f"✓ OpenAI 套件版本: {openai.__version__}")
    except ImportError:
        print("⚠ 警告: 未安裝 openai 套件")
        print("請執行: pip install openai")
        return
    
    if task == 'B1':
        run_b1()
    elif task == 'B2':
        run_b2()
    elif task == 'B3':
        run_b3()
    elif task == 'ALL' or task is None:
        run_b1()
        run_b2()
        run_b3()
    else:
        print(f"⚠ 無效的任務: {task}")
        return
    
    print("\n" + "="*60)
    print("🎉 Part B 完成！")
    print("="*60)
    print("\n結果檔案位於 results/ 資料夾")


if __name__ == "__main__":
    # 支援命令列參數
    if len(sys.argv) > 1:
        task = sys.argv[1].upper()
        main(task)
    else:
        main('ALL')