#!/usr/bin/env python3
"""
1bit LLMを使用したコード生成の例
"""

import sys
import os

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codegen import BitLLMCodeGenerator


def main():
    """例: Python関数の生成"""
    
    # コード生成器を初期化
    generator = BitLLMCodeGenerator()
    
    # 例1: 単純なPython関数
    print("=" * 60)
    print("例1: 2つの数字を足す関数")
    print("=" * 60)
    code1 = generator.generate_code(
        prompt="2つの数字を足すPython関数を作成して",
        language="python",
        max_tokens=100
    )
    print(code1)
    print()
    
    # 例2: クラスの生成
    print("=" * 60)
    print("例2: 簡単なCalculatorクラス")
    print("=" * 60)
    code2 = generator.generate_code(
        prompt="加算、減算、乗算、除算ができるCalculatorクラスを作成して",
        language="python",
        max_tokens=200
    )
    print(code2)
    print()
    
    # 例3: JavaScript関数
    print("=" * 60)
    print("例3: JavaScriptの配列操作関数")
    print("=" * 60)
    code3 = generator.generate_code(
        prompt="配列の要素を2倍にするJavaScript関数を作成して",
        language="javascript",
        max_tokens=100
    )
    print(code3)
    print()
    
    # 例4: TypeScriptの型定義
    print("=" * 60)
    print("例4: TypeScriptのインターフェース")
    print("=" * 60)
    code4 = generator.generate_code(
        prompt="User型のTypeScriptインターフェースを作成して、name、age、emailのプロパティを含めて",
        language="typescript",
        max_tokens=100
    )
    print(code4)


if __name__ == "__main__":
    main()
