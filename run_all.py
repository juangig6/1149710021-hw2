#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作業2 主執行程式
支援執行各個任務或完整作業
"""
import os
import sys
from datetime import datetime

def print_header(title):
    """列印標題"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def show_part_a_menu():
    """顯示 Part A 子選單"""
    while True:
        print_header("Part A - 傳統方法")
        print("\nPart A 子選單:")
        print("  A1. TF-IDF 文本相似度計算")
        print("  A2. 基於規則的文本分類")
        print("  A3. 統計式自動摘要")
        print("  A0. 執行完整 Part A (A1 + A2 + A3)")
        print("  0.  返回主選單")
        
        choice = input("\n請輸入選項: ").strip().upper()
        
        if choice in ['A1', 'A2', 'A3', 'A0']:
            print_header(f"執行 {choice if choice != 'A0' else 'Part A 完整'}")
            try:
                import traditional_methods
                if choice == 'A1':
                    traditional_methods.run_a1()
                elif choice == 'A2':
                    traditional_methods.run_a2()
                elif choice == 'A3':
                    traditional_methods.run_a3()
                elif choice == 'A0':
                    traditional_methods.main('ALL')
            except Exception as e:
                print(f"⚠ 執行錯誤: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == '0':
            break
        else:
            print("\n⚠ 無效的選項，請重新輸入")
        
        if choice != '0':
            input("\n按 Enter 繼續...")

def show_part_b_menu():
    """顯示 Part B 子選單"""
    while True:
        print_header("Part B - AI 方法")
        print("\nPart B 子選單:")
        print("  B1. 語意相似度計算")
        print("  B2. AI 文本分類")
        print("  B3. AI 自動摘要")
        print("  B0. 執行完整 Part B (B1 + B2 + B3)")
        print("  0.  返回主選單")
        
        choice = input("\n請輸入選項: ").strip().upper()
        
        if choice in ['B1', 'B2', 'B3', 'B0']:
            print_header(f"執行 {choice if choice != 'B0' else 'Part B 完整'}")
            try:
                import modern_methods
                if choice == 'B1':
                    modern_methods.run_b1()
                elif choice == 'B2':
                    modern_methods.run_b2()
                elif choice == 'B3':
                    modern_methods.run_b3()
                elif choice == 'B0':
                    modern_methods.main('ALL')
            except ImportError:
                print("⚠ modern_methods.py 尚未實作")
            except Exception as e:
                print(f"⚠ 執行錯誤: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == '0':
            break
        else:
            print("\n⚠ 無效的選項，請重新輸入")
        
        if choice != '0':
            input("\n按 Enter 繼續...")

def show_part_c_menu():
    """顯示 Part C 子選單"""
    while True:
        print_header("Part C - 比較分析")
        print("\nPart C 子選單:")
        print("  C1. 量化比較")
        print("  C2. 質性分析")
        print("  C0. 執行完整 Part C (C1 + C2)")
        print("  0.  返回主選單")
        
        choice = input("\n請輸入選項: ").strip().upper()
        
        if choice in ['C1', 'C2', 'C0']:
            print_header(f"執行 {choice if choice != 'C0' else 'Part C 完整'}")
            try:
                import comparison
                if choice == 'C1':
                    comparison.run_c1()
                elif choice == 'C2':
                    comparison.run_c2()
                elif choice == 'C0':
                    comparison.main('ALL')
            except ImportError:
                print("⚠ comparison.py 尚未實作")
            except Exception as e:
                print(f"⚠ 執行錯誤: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == '0':
            break
        else:
            print("\n⚠ 無效的選項，請重新輸入")
        
        if choice != '0':
            input("\n按 Enter 繼續...")

def run_all_parts():
    """執行完整作業 (Part A + B + C)"""
    print_header("執行完整作業 (Part A + B + C)")
    
    # Part A
    print("\n" + "="*60)
    print("開始執行 Part A...")
    print("="*60)
    try:
        import traditional_methods
        traditional_methods.main('ALL')
    except Exception as e:
        print(f"⚠ Part A 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    # Part B
    print("\n" + "="*60)
    print("開始執行 Part B...")
    print("="*60)
    try:
        import modern_methods
        modern_methods.main('ALL')
    except ImportError:
        print("⚠ modern_methods.py 尚未實作，跳過 Part B")
    except Exception as e:
        print(f"⚠ Part B 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    # Part C
    print("\n" + "="*60)
    print("開始執行 Part C...")
    print("="*60)
    try:
        import comparison
        comparison.main('ALL')
    except ImportError:
        print("⚠ comparison.py 尚未實作，跳過 Part C")
    except Exception as e:
        print(f"⚠ Part C 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主程式"""
    print_header("作業2 - 文本處理方法實作與比較")
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        print("\n" + "="*60)
        print("主選單:")
        print("="*60)
        print("1. 執行完整作業 (Part A + B + C)")
        print("2. Part A - 傳統方法 (含子選單)")
        print("3. Part B - AI方法 (含子選單)")
        print("4. Part C - 比較分析 (含子選單)")
        print("5. 離開")
        
        choice = input("\n請輸入選項 (1-5): ").strip()
        
        if choice == '1':
            run_all_parts()
            print("\n" + "="*60)
            print("🎉 完整作業執行完成！")
            print("="*60)
            print("\n結果檔案位於 results/ 資料夾")
            input("\n按 Enter 繼續...")
            
        elif choice == '2':
            show_part_a_menu()
            
        elif choice == '3':
            show_part_b_menu()
            
        elif choice == '4':
            show_part_c_menu()
            
        elif choice == '5':
            print("\n再見！")
            sys.exit(0)
            
        else:
            print("\n⚠ 無效的選項，請重新輸入")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程式被使用者中斷")
        sys.exit(0)
    except Exception as e:
        print(f"\n⚠ 程式執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)