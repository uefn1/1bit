# 4bit LLM モデル選定

> **注**: 1bit量子化はBitNet専用です。一般的なLLMモデル（LLaMA, Qwen, phi等）では**4bit量子化**を使用します。

## 🎯 選定結果

**選定モデル: [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B) + 4bit量子化**

### 理由
- **GitHub Actions環境** (2 vCPU, 7GB RAM) で動作可能
- **4bit量子化** (`load_in_4bit=True`) によりメモリ使用量が大幅に削減
- **TinyLlama-1.1B** (1.1Bパラメータ) は軽量でコード生成に適切
- **フォールバック機能**により、モデルが見つからない場合に自動的に代替モデルを試行

---

## 📊 選定基準

### 要件
| 項目 | 要件 |
|------|------|
| 用途 | コード生成 |
| 実行環境 | GitHub Actions (CPU, 7GB RAM) |
| モデルタイプ | 4bit量子化LLM (1bitはBitNet専用) |

### 候補比較

| モデル | パラメータ | FP16サイズ | 4bitサイズ | 推論メモリ | コード生成性能 | GitHub Actions動作 | 推奨度 |
|--------|------------|------------|------------|--------------|----------------|-------------------|--------|
| BitNet b1.58 | 58B | ~116GB | ~3.6GB | ~6-8GB | ⭐⭐⭐⭐⭐ | ❌ (メモリ不足) | - |
| phi-2 | 2.7B | ~5.4GB | ~1.4GB | ~2-3GB | ⭐⭐⭐⭐⭐ | ⚠️ (7GBギリギリ) | ⭐⭐⭐ |
| **TinyLlama-1.1B** | 1.1B | ~2.2GB | **~560MB** | **~1-2GB** | ⭐⭐⭐⭐ | **✅** | **⭐⭐⭐⭐⭐** |
| Qwen2-0.5B | 0.5B | ~1GB | ~250MB | ~500MB-1GB | ⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| Gemma-2B | 2B | ~4GB | ~1GB | ~2-3GB | ⭐⭐⭐⭐ | ⚠️ (7GBギリギリ) | ⭐⭐⭐ |

> **注**: 4bit量子化により、モデルサイズは元のFP16から1/4に圧縮されます

---

## 🏆 選定モデル詳細

### TinyLlama-1.1B + 4bit量子化

**仕様**
- アーキテクチャ: LLaMA
- パラメータ: 1.1B
- 量子化: 4bit (NF4量子化, `load_in_4bit=True`)
- FP16モデルサイズ: ~2.2GB → **4bit化で~560MB**
- 推論時メモリ: ~1-2GB (KVキャッシュ含む)

**特徴**
- ✅ GitHub Actionsの7GB RAM内で余裕を持って動作
- ✅ CPU上でも推論可能 (若干遅いが実用的)
- ✅ コード生成タスクに適切な性能
- ✅ HuggingFace Transformers + bitsandbytesで利用可能
- ✅ 4bit量子化により高速な推論

**推奨使用シナリオ**
- GitHub Actionsでの自動コード生成
- CI/CDパイプラインでのコード補完
- 軽量なコードレビューアシスタント

---

## 💼 代替モデル候補

### 第2候補: Qwen2-0.5B-Instruct + 4bit
- **0.5Bパラメータ** - 非常に軽量
- 4bitサイズ: ~250MB
- 推論メモリ: ~500MB-1GB
- 高速な推論
- 精度はやや劣る

### 第3候補: microsoft/phi-2 + 4bit
- **2.7Bパラメータ** - 中規模
- 4bitサイズ: ~1.4GB
- 推論メモリ: ~2-3GB
- 高いコード生成性能
- GitHub Actionsの7GB制限内で動作

### 第4候補: google/gemma-2b + 4bit
- **2Bパラメータ**
- 4bitサイズ: ~1GB
- 推論メモリ: ~2-3GB
- Googleによる高品質モデル

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

### 2. HuggingFace Hub認証 (オプショナル)

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

## ⚠️ 注意事項

1. **GitHub Actionsのメモリ制限**: 7GBを超える場合はモデルサイズを小さくしてください
2. **推論速度**: CPU上での推論はGPUより遅いです (7Bモデルで数秒〜数十秒)
3. **モデル精度**: 4bit量子化により、若干の精度低下があります
4. **ネットワーク**: 初回実行時はモデルダウンロードに時間がかかります (数分〜10分)
5. **HuggingFace Hub認証**: プライベートモデルを使用する場合は、`huggingface-cli login`で認証してください
6. **1bit量子化**: 1bit量子化は**BitNet専用**です。一般的なモデルでは4bitを使用します。

---

## 📚 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2402.17764)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [HuggingFace 4bit Quantization](https://huggingface.co/docs/transformers/main_classes/quantization)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [TinyLlama on HuggingFace](https://huggingface.co/TinyLlama)
- [Qwen2 on HuggingFace](https://huggingface.co/Qwen)
- [Phi-2 on HuggingFace](https://huggingface.co/microsoft/phi-2)

---

## 🔍 トラブルシューティング

### エラー: `RepositoryNotFoundError`
**原因**: モデルIDが間違っている、またはモデルが存在しない
**解決**: 
- `config.yaml`の`huggingface_id`を正しいモデルIDに更新
- 利用可能なモデル: `TinyLlama/TinyLlama-1.1B`, `Qwen/Qwen2-0.5B-Instruct`, `microsoft/phi-2`

### エラー: `load_in_4bit is not supported`
**原因**: bitsandbytesのバージョンが古い
**解決**: 
```bash
pip install --upgrade bitsandbytes transformers accelerate
```

### エラー: メモリ不足
**原因**: モデルが大きすぎる
**解決**: 
- より小規模なモデルを使用 (Qwen2-0.5B, TinyLlama-1.1B)
- 8bit量子化を使用 (`load_in_8bit=True`)

### エラー: 遅すぎる
**原因**: CPU上での推論は遅い
**解決**: 
- より小規模なモデルを使用
- GitHub Actionsの有料ランナー (GPU) を使用
