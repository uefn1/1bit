# 1bit LLM モデル選定

## 🎯 選定結果

**選定モデル: [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B) + 1bit量子化**

### 理由
- **GitHub Actions環境** (2 vCPU, 7GB RAM) で動作可能
- **1bit量子化** (`load_in_1bit=True`) によりメモリ使用量が大幅に削減
- **TinyLlama-1.1B** (1.1Bパラメータ) は軽量でコード生成に適切
- **フォールバック機能**により、モデルが見つからない場合に自動的に代替モデルを試行

---

## 📊 選定基準

### 要件
| 項目 | 要件 |
|------|------|
| 用途 | コード生成 |
| 実行環境 | GitHub Actions (CPU, 7GB RAM) |
| モデルタイプ | 1bit量子化LLM |

### 候補比較

| モデル | パラメータ | FP16サイズ | 1bitサイズ | 推論メモリ | コード生成性能 | GitHub Actions動作 | 推奨度 |
|--------|------------|------------|------------|--------------|----------------|-------------------|--------|
| BitNet b1.58 | 58B | ~116GB | ~3.6GB | ~6-8GB | ⭐⭐⭐⭐⭐ | ❌ (メモリ不足) | - |
| phi-2 | 2.7B | ~5.4GB | ~170MB | ~800MB-1GB | ⭐⭐⭐⭐⭐ | ⚠️ (7GBギリギリ) | ⭐⭐⭐ |
| **TinyLlama-1.1B** | 1.1B | ~2.2GB | **~70MB** | **~200-400MB** | ⭐⭐⭐⭐ | **✅** | **⭐⭐⭐⭐⭐** |
| Qwen2-0.5B | 0.5B | ~1GB | ~30MB | ~100-200MB | ⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| TinyLLaMA | 1B | ~2GB | ~60MB | ~150-300MB | ⭐⭐⭐ | ✅ | ⭐⭐⭐ |

> **注**: 1bit量子化により、モデルサイズは元の16bit浮動小数点から1/32に圧縮されます

---

## 🏆 選定モデル詳細

### TinyLlama-1.1B + 1bit量子化

**仕様**
- アーキテクチャ: LLaMA
- パラメータ: 1.1B
- 量子化: 1bit (重み、`load_in_1bit=True`で適用)
- FP16モデルサイズ: ~2.2GB → **1bit化で~70MB**
- 推論時メモリ: ~200-400MB (KVキャッシュ含む)

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

## 💼 代替モデル候補

### 第2候補: Qwen2-0.5B + 1bit
- **0.5Bパラメータ** - 非常に軽量
- 推論メモリ: ~100-200MB
- 高速な推論
- 精度はやや劣る

### 第3候補: microsoft/phi-2 + 1bit
- **2.7Bパラメータ** - 中規模
- 推論メモリ: ~800MB-1GB
- 高いコード生成性能
- GitHub Actionsの7GB制限内で動作 (ギリギリ)

### 第4候補: BitNet (Microsoft公式)
- **本格的な1bitモデル**
- 精度が高い
- 実装が複雑
- 大規模モデル (58B) はGitHub Actionsで動作しない

---

## 📁 リポジトリ構成

```
1bit/
├── README.md          # このファイル
├── requirements.txt   # 依存関係
├── codegen.py         # コード生成スクリプト
├── config.yaml        # モデル設定
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

### 2. モデルのダウンロード

初回実行時に自動的にHuggingFace Hubからダウンロードされます。

**フォールバック機能**: モデルが見つからない場合、自動的に以下のモデルを試行します:
- `TinyLlama/TinyLlama-1.1B` (デフォルト)
- `Qwen/Qwen2-0.5B`
- `microsoft/phi-2`

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

### GitHub Actionsでの実行

ワークフローが自動的にトリガーされ、コード生成タスクを実行します。

---

## ⚠️ 注意事項

1. **GitHub Actionsのメモリ制限**: 7GBを超える場合はモデルサイズを小さくしてください
2. **推論速度**: CPU上での推論はGPUより遅いです
3. **モデル精度**: 1bit量子化により、若干の精度低下があります
4. **ネットワーク**: 初回実行時はモデルダウンロードに時間がかかります
5. **HuggingFace Hub認証**: プライベートモデルを使用する場合は、`huggingface-cli login`で認証してください

---

## 📚 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2402.17764)
- [BiLLM: Binary Large Language Models](https://arxiv.org/abs/2403.09364)
- [HuggingFace bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [TinyLlama on HuggingFace](https://huggingface.co/TinyLlama/TinyLlama-1.1B)
