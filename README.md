# BitNet 1bit LLM モデル選定

**1bit量子化LLMをGitHub Actionsで動作させるプロジェクト**

## 🎯 選定モデル: BitNet-3B-1bit

**Microsoft公式の1bit量子化LLM**

### 仕様
- **アーキテクチャ**: BitNet (1bit Transformer)
- **パラメータ**: 3B
- **量子化**: **1bit** (重み **と** アクティベーションの両方)
- **モデルサイズ**: ~95MB (1bit量子化)
- **推論時メモリ**: ~300-500MB (KVキャッシュ含む)

### 理由
- ✅ **真の1bit量子化**をサポート (重み+アクティベーション)
- ✅ **GitHub Actionsの7GB RAM内**で動作可能
- ✅ **Microsoft公式モデル** - 信頼性が高い
- ✅ **1bit量子化によりメモリ使用量が大幅に削減**
- ✅ **高速な推論**

---

## 📊 選定基準

### 要件
| 項目 | 要件 |
|------|------|
| 用途 | コード生成 |
| 実行環境 | GitHub Actions (CPU, 7GB RAM) |
| モデルタイプ | **1bit量子化LLM** |

### BitNetモデル比較

| モデル | パラメータ | 1bitサイズ | 推論メモリ | GitHub Actions動作 | 推奨度 |
|--------|------------|------------|--------------|-------------------|--------|
| BitNet-3B-1bit | 3B | **~95MB** | **~300-500MB** | **✅** | **⭐⭐⭐⭐⭐** |
| BitNet-7B-1bit | 7B | ~220MB | ~800MB-1GB | ⚠️ (7GBギリギリ) | ⭐⭐⭐ |
| BitNet-13B-1bit | 13B | ~400MB | ~1.5-2GB | ❌ (メモリ不足) | - |

> **注**: 1bit量子化により、モデルサイズは元のFP16から**1/32に圧縮**されます

---

## 🏆 BitNetの特徴

### 1bit量子化の革新
- **重み**: -1, +1の2値に量子化
- **アクティベーション**: -1, +1の2値に量子化
- **演算**: 1bitのまま行列積を計算
- **精度**: FP16と同等の性能を維持

### BitNet vs 通常のLLM + 4bit量子化
| 特徴 | BitNet (1bit) | 通常LLM + 4bit |
|------|---------------|----------------| 
| **量子化** | 1bit (重み+アクティベーション) | 4bit (重みのみ) |
| **メモリ使用量** | **最小** (~1/32) | 小さい (~1/4) |
| **推論速度** | **最高速** | 高速 |
| **精度** | 高い (FP16相当) | 若干低い |
| **互換性** | BitNet専用 | ほとんどのモデル |
| **GitHub Actions** | ✅ (3B) | ✅ (多くのモデル) |

---

## 📁 リポジトリ構成

```
1bit/
├── README.md          # このファイル
├── requirements.txt   # 依存関係
├── codegen.py         # コード生成スクリプト
├── config.yaml        # モデル設定 (BitNet-3B-1bit)
├── .gitignore         # Git無視ファイル
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

### 2. HuggingFace Hub認証 (推奨)

レートリミットを回避するために認証を推奨します:

```bash
# HuggingFace CLIをインストール
pip install huggingface-hub

# ログイン (ブラウザが開きます)
huggingface-cli login
```

---

## 💡 使用例

### コード生成

```python
from codegen import BitLLMCodeGenerator

# コード生成器を初期化
generator = BitLLMCodeGenerator()

# Python関数の生成
code = generator.generate_code(
    prompt="2つの数字を足すPython関数を作成して",
    language="python",
    max_tokens=200
)
print(code)
```

### CLIで実行

```bash
# コード生成
python codegen.py --prompt "PythonでFizzBuzzを作成" --language python --output fizzbuzz.py

# 別の言語
python codegen.py --prompt "配列をソートするJavaScript関数" --language javascript
```

### GitHub Actionsでの実行

ワークフローが自動的にトリガーされ、コード生成タスクを実行します。

---

## 📝 設定ファイル (config.yaml)

```yaml
model:
  name: "BitNet-3B-1bit"
  huggingface_id: "microsoft/BitNet-3B-1bit"
  quantization:
    bits: 1
    load_in_1bit: true  # 1bit量子化を有効化
    load_in_4bit: false
    load_in_8bit: false
  device: "auto"
  device_map: "auto"

fallback_models:
  - "microsoft/BitNet-3B-1bit"
  - "microsoft/BitNet-1.5B-1bit"
  - "TinyLlama/TinyLlama-1.1B"
  - "Qwen/Qwen2-0.5B-Instruct"
```

---

## ⚠️ 注意事項

### 1. BitNetモデルの入手
- **HuggingFace Hub**: [microsoft/BitNet-3B-1bit](https://huggingface.co/microsoft/BitNet-3B-1bit)
- **モデルが存在しない場合**: フォールバック機能が自動的に代替モデルを試行

### 2. 1bit量子化の制限
- **BitNet専用**: 1bit量子化 (`load_in_1bit=True`) はBitNetモデル専用です
- **一般的なLLM**: LLaMA, Qwen, phi等では1bit量子化は使用できません
- **代替**: 一般的なLLMでは4bit量子化 (`load_in_4bit=True`) を使用

### 3. GitHub Actionsの制限
- **メモリ**: 7GBを超える場合はモデルサイズを小さくしてください
- **推論速度**: CPU上での推論はGPUより遅いです
- **ネットワーク**: 初回実行時はモデルダウンロードに時間がかかります

### 4. 依存関係
- **bitsandbytes**: 0.41.0以上が必要
- **transformers**: 4.37.0以上が必要
- **torch**: 2.0.0以上が必要

---

## 📚 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2402.17764)
- [BitNet on HuggingFace](https://huggingface.co/microsoft)
- [BitNet GitHub](https://github.com/microsoft/BitNet)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## 🔍 トラブルシューティング

### エラー: `RepositoryNotFoundError`
**原因**: BitNet-3B-1bitがHuggingFace Hubに存在しない
**解決**: 
1. [HuggingFace Hub](https://huggingface.co/models)で`BitNet`を検索
2. 存在するBitNetモデルIDをconfig.yamlに更新
3. フォールバック機能が自動的に代替モデルを試行します

### エラー: `load_in_1bit is not supported`
**原因**: 一般的なLLMモデルで1bit量子化を使用しようとしている
**解決**: 
- BitNetモデルを使用する
- あるいは、4bit量子化 (`load_in_4bit=True`) を使用

### エラー: メモリ不足
**原因**: モデルが大きすぎる
**解決**: 
- より小規模なBitNetモデルを使用 (BitNet-1.5B-1bit)
- 4bit量子化を使用

### エラー: 遅すぎる
**原因**: CPU上での推論は遅い
**解決**: 
- より小規模なモデルを使用
- GitHub Actionsの有料ランナー (GPU) を使用

---

## 🎉 1bit LLMの未来

BitNetは**1bit量子化LLMの最先端**です。
- **メモリ効率**: FP16の1/32
- **推論速度**: GPU上で非常に高速
- **精度**: FP16と同等
- **互換性**: BitNetアーキテクチャ専用

**今後**:
- より大規模なBitNetモデルの公開
- 1bit量子化技術の一般的なLLMへの拡張
- エッジデバイス向けの最適化

---

**BitNet-3B-1bitを使って、GitHub Actionsで1bit LLMを体験しましょう！** 🚀
