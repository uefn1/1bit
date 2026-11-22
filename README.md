# BitNet b1.58-2B-4T + bitnet.cpp

**BitNet 1bit量子化LLMをbitnet.cpp (C++推論エンジン) で使用するプロジェクト**

## 🎯 選定モデル: BitNet b1.58-2B-4T

**Microsoft公式の1bit量子化LLM + 高速C++推論エンジン**

### 仕様
| 項目 | 内容 |
|------|------|
| **モデル** | BitNet b1.58-2B-4T |
| **パラメータ** | 2B |
| **量子化** | **1bit** (重み **と** アクティベーション) |
| **推論エンジン** | **bitnet.cpp** (C++高速エンジン) |
| **モデルフォーマット** | GGUF |
| **推論メモリ** | ~500MB-1GB |
| **GitHub Actions** | ✅ **動作可能** (7GB内) |

---

## 🚀 なぜbitnet.cppを使用するのか？

### bitnet.cppの利点
1. **高速推論**: C++で書かれた高速な推論エンジン
2. **低メモリ**: 1bit量子化によりメモリ使用量が最小
3. **CPU最適化**: CPU上で高速に動作
4. **GGUFフォーマット**: 量子化済みのモデルを直接使用
5. **BitNet専用**: BitNetアーキテクチャに最適化

### HuggingFace Transformersとの比較
| 特徴 | bitnet.cpp | Transformers |
|------|------------|-------------|
| **言語** | C++ | Python |
| **速度** | ⭐⭐⭐⭐⭐ **最高速** | ⭐⭐⭐ 高速 |
| **メモリ** | ⭐⭐⭐⭐⭐ **最小** | ⭐⭐⭐⭐ 小さい |
| **GPUサポート** | ❌ CPU専用 | ✅ GPU/CPU |
| **1bitサポート** | ✅ **完全サポート** | ⚠️ BitNet専用 |
| **セットアップ** | ⚠️ ビルド必要 | ✅ 簡単 |

---

## 📁 リポジトリ構成

```
1bit/
├── README.md              # このファイル
├── requirements.txt       # Python依存関係
├── codegen.py            # コード生成スクリプト (bitnet.cpp版)
├── config.yaml           # モデル設定 (BitNet b1.58-2B-4T)
├── .gitignore            # Git無視ファイル
├── .github/
│   └── workflows/
│       └── codegen.yml  # GitHub Actionsワークフロー
└── bitnet/               # bitnet.cppリポジトリ (サブディレクトリ)
    ├── CMakeLists.txt
    ├── run_inference.py
    ├── src/
    │   └── ...          # C++ソースコード
    └── models/           # GGUFモデルを配置
```

---

## 🔧 セットアップ

### 1. 依存関係のインストール

```bash
# Python依存関係
pip install -r requirements.txt

# bitnet.cppの依存関係 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

### 2. bitnet.cppリポジトリをクローン

```bash
# 既にクローンされている場合はスキップ
cd /path/to/this/repo
git clone https://github.com/microsoft/BitNet.git bitnet
```

### 3. bitnet.cppをビルド

```bash
cd bitnet
mkdir build
cd build
cmake ..
make -j$(nproc)
```

**ビルド時間**: 5-10分 (GitHub Actionsでは10-15分)

### 4. GGUFモデルをダウンロード

```bash
# HuggingFace HubからGGUFモデルをダウンロード
# 注: 現在、BitNetのGGUFモデルは公開され次第更新します

# 手動ダウンロードの場合
# 1. https://huggingface.co/models?search=bitnet にアクセス
# 2. BitNet-b1.58-2B-4TのGGUFファイルをダウンロード
# 3. bitnet/models/bitnet-b1.58-2B-4T/ に配置
```

**GGUFモデルの例:**
- `ggml-model-i2_s.gguf` (1bit量子化)
- `ggml-model-f16.gguf` (FP16)

---

## 💡 使用例

### 1. コード生成器を使用

```python
from codegen import BitNetCodeGenerator

# コード生成器を初期化
generator = BitNetCodeGenerator()

# Python関数の生成
code = generator.generate_code(
    prompt="2つの数字を足すPython関数を作成して",
    language="python",
    max_tokens=200
)
print(code)
```

### 2. CLIで実行

```bash
# コード生成
python codegen.py --prompt "PythonでFizzBuzzを作成" --language python --output fizzbuzz.py

# 別の言語
python codegen.py --prompt "配列をソートするJavaScript関数" --language javascript
```

### 3. bitnet.cppを直接使用

```bash
# bitnet.cppのrun_inference.pyを直接使用
cd bitnet
python run_inference.py \
    -m models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "PythonでHello Worldを作成" \
    -n 100 \
    --temp 0.7
```

### 4. llama-cliを直接使用

```bash
# ビルドしたllama-cliを直接使用
./bitnet/build/bin/llama-cli \
    -m models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "PythonでHello Worldを作成" \
    -n 100 \
    --temp 0.7
```

---

## 📝 設定ファイル (config.yaml)

```yaml
model:
  name: "BitNet-b1.58-2B-4T"
  huggingface_id: "microsoft/bitnet-b1.58-2B-4T"
  gguf_model: "bitnet-b1.58-2B-4T.gguf"
  quantization:
    bits: 1
    type: "i2_s"  # BitNetの量子化タイプ
  engine: "bitnet.cpp"  # 推論エンジン
  
bitnet_cpp:
  use_gguf: true
  model_path: "bitnet/models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"
  threads: 2
  ctx_size: 2048
```

---

## 🏗️ GitHub Actionsワークフロー

GitHub Actionsでbitnet.cppを使用する場合:

1. **ビルド**: bitnet.cppを自動ビルド
2. **モデルダウンロード**: GGUFモデルをダウンロード
3. **推論**: コード生成を実行

**ワークフロー**: [.github/workflows/codegen.yml](.github/workflows/codegen.yml)

### 注意点
- **ビルド時間**: 10-15分 (GitHub Actionsの無料ランナー)
- **メモリ**: 7GB内で動作 (2Bモデル)
- **キャッシュ**: ビルド結果をキャッシュして再ビルドを回避

---

## 📊 BitNet b1.58-2B-4Tの特徴

### 1bit量子化の革新
- **重み**: -1, +1の2値に量子化
- **アクティベーション**: -1, +1の2値に量子化
- **演算**: 1bitのまま行列積を計算
- **精度**: FP16と同等の性能を維持

### b1.58の意味
- **b1.58**: 1.58ビットの平均量子化 (1bit + 0.58bitの補正)
- **2B**: 20億パラメータ
- **4T**: 4兆トークンで学習

### 推論性能
| 環境 | 推論速度 | メモリ使用量 |
|------|----------|--------------|
| CPU (2コア) | ~10-20 tokens/sec | ~500MB-1GB |
| CPU (4コア) | ~20-40 tokens/sec | ~500MB-1GB |
| CPU (8コア) | ~40-80 tokens/sec | ~500MB-1GB |

---

## ⚠️ 注意事項

### 1. bitnet.cppのビルド
- **時間**: 5-15分 (マシンによって異なる)
- **ディスク容量**: ~2-3GB (ビルド中)
- **依存関係**: CMake, build-essential, git

### 2. GGUFモデルの入手
- **現状**: BitNetのGGUFモデルはまだHuggingFace Hubに公開されていない可能性があります
- **代替**: HuggingFace Transformers + `load_in_1bit=True` (BitNetモデル専用)

### 3. GitHub Actionsの制限
- **ビルド時間**: 10-15分 (無料ランナー)
- **メモリ**: 7GB内で動作 (2Bモデル)
- **タイムアウト**: 60分に設定 (ビルド+推論)

### 4. 1bit量子化の制限
- **BitNet専用**: 1bit量子化 (`load_in_1bit=True`) はBitNetモデル専用
- **一般的なLLM**: LLaMA, Qwen, phi等では1bit量子化は使用できません

---

## 📚 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2402.17764)
- [BitNet GitHub](https://github.com/microsoft/BitNet)
- [bitnet.cpp GitHub](https://github.com/microsoft/BitNet) (同一リポジトリ)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## 🔍 トラブルシューティング

### エラー: `bitnet.cpp directory not found`
**原因**: bitnet.cppリポジトリがクローンされていない
**解決**: 
```bash
cd /path/to/this/repo
git clone https://github.com/microsoft/BitNet.git bitnet
```

### エラー: `llama-cli not found`
**原因**: bitnet.cppがビルドされていない
**解決**: 
```bash
cd bitnet
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### エラー: `GGUF model not found`
**原因**: GGUFモデルが存在しない
**解決**: 
1. [HuggingFace Hub](https://huggingface.co/models?search=bitnet)でBitNetモデルを検索
2. GGUFファイルをダウンロード
3. `bitnet/models/bitnet-b1.58-2B-4T/` に配置

### エラー: `build failed`
**原因**: 依存関係が不足
**解決**: 
```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential cmake git

# Mac
brew install cmake
```

### エラー: `Out of memory`
**原因**: モデルが大きすぎる
**解決**: 
- より小規模なBitNetモデルを使用 (1.5B, 3B)
- `threads`パラメータを減らす

---

## 🎉 1bit LLM + C++推論エンジン

BitNet b1.58-2B-4T + bitnet.cppの組み合わせは:
- ✅ **真の1bit量子化** (重み+アクティベーション)
- ✅ **高速推論** (C++エンジン)
- ✅ **低メモリ** (~500MB-1GB)
- ✅ **GitHub Actions対応** (7GB内で動作)

**次世代の1bit LLMを体験しましょう！** 🚀
