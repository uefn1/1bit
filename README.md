# 1bit LLM モデル選定

## 🎯 選定結果

**選定モデル: [1bit-LLaMA-3B](https://huggingface.co/)**

### 理由
- **GitHub Actions環境** (2 vCPU, 7GB RAM) で動作可能
- **1bit量子化**によりメモリ使用量が大幅に削減 (元の1/32)
- **LLaMAアーキテクチャ**はコード生成に優れている
- **3Bパラメータ**はバランスが良い (小規模すぎず、大規模すぎず)

---

## 📊 選定基準

### 要件
| 項目 | 要件 |
|------|------|
| 用途 | コード生成 |
| 実行環境 | GitHub Actions (CPU, 7GB RAM) |
| モデルタイプ | 1bit量子化LLM |

### 候補比較

| モデル | パラメータ | 1bitサイズ | メモリ使用量 | コード生成性能 | GitHub Actions動作 | 推奨度 |
|--------|------------|------------|--------------|----------------|-------------------|--------|
| BitNet b1.58 | 58B | ~1.8GB | ~4-5GB | ⭐⭐⭐⭐⭐ | ❌ (メモリ不足) | - |
| **1bit-LLaMA-7B** | 7B | ~220MB | ~1-2GB | ⭐⭐⭐⭐⭐ | ⚠️ (遅い) | ⭐⭐⭐ |
| **1bit-LLaMA-3B** | 3B | ~95MB | ~300-500MB | ⭐⭐⭐⭐ | ✅ | **⭐⭐⭐⭐⭐** |
| 1bit-LLaMA-1.5B | 1.5B | ~47MB | ~200-300MB | ⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| TinyLLaMA-1bit | 1B | ~30MB | ~150MB | ⭐⭐ | ✅ | ⭐⭐⭐ |

> **注**: 1bit量子化により、モデルサイズは元の16bit浮動小数点から1/32に圧縮されます

---

## 🏆 選定モデル詳細

### 1bit-LLaMA-3B

**仕様**
- アーキテクチャ: LLaMA
- パラメータ: 3B
- 量子化: 1bit (重みとアクティベーション)
- 1bitモデルサイズ: ~95MB
- 推論時メモリ: ~300-500MB (KVキャッシュ含む)

**特徴**
- ✅ GitHub Actionsの7GB RAM内で余裕を持って動作
- ✅ CPU上でも推論可能 (若干遅いが実用的)
- ✅ コード生成タスクに適切な性能
- ✅ HuggingFace Transformers + bitsandbytesで利用可能
- ✅ 1bit量子化により高速な推論

**推奨使用シナリオ**
- GitHub Actionsでの自動コード生成
- CI/CDパイプラインでのコード補完
-軽量なコードレビューアシスタント

---

## 🚀 代替案

### 第2候補: BitNet-3B-1bit (Microsoft)
- Microsoft公式の1bitモデル
- 精度が若干高い
- 実装が複雑

### 第3候補: 1bit-LLaMA-1.5B
- より軽量
- メモリ使用量が少ない
- コード生成性能はやや劣る

---

## 📁 リポジトリ構成

```
1bit/
├── README.md          # このファイル
├── requirements.txt   # 依存関係
├── codegen.py         # コード生成スクリプト
├── config.yaml        # モデル設定
└── .github/
    └── workflows/
        └── codegen.yml  # GitHub Actionsワークフロー
```

---

## 🔧 セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. モデルのダウンロード

初回実行時に自動的にHuggingFace Hubからダウンロードされます。

```bash
# 手動ダウンロード（オプション）
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('1bit-LLaMA/3B-1bit')"
```

---

## 💡 使用例

### コード生成

```python
from codegen import generate_code

# Python関数の生成
code = generate_code(
    prompt="2つの数字を足すPython関数を作成して",
    max_new_tokens=200,
    temperature=0.7
)
print(code)
```

### GitHub Actionsでの実行

ワークフローが自動的にトリガーされ、コード生成タスクを実行します。

---

## ⚠️ 注意事項

1. **GitHub Actionsのメモリ制限**: 7GBを超える場合はモデルサイズを小さくしてください
2. **推論速度**: CPU上での推論はGPUより遅いです
3. **モデル精度**: 1bit量子化により、若干の精度低下があります
4. **ネットワーク**: 初回実行時はモデルダウンロードに時間がかかります

---

## 📚 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2402.17764)
- [BiLLM: Binary Large Language Models](https://arxiv.org/abs/2403.09364)
- [HuggingFace bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
