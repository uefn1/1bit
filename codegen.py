#!/usr/bin/env python3
"""
BitNet b1.58-2B-4T コード生成スクリプト (bitnet.cppを使用)

GitHub Actionsで動作するBitNet 1bit量子化LLMを使用したコード生成
bitnet.cpp (C++推論エンジン) を使用します
"""

import os
import sys
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any


class BitNetCodeGenerator:
    """BitNet b1.58-2B-4Tを使用したコード生成器 (bitnet.cpp版)"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        コード生成器を初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self._setup_bitnet_cpp()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_bitnet_cpp(self):
        """bitnet.cppの環境をセットアップ"""
        print("Setting up bitnet.cpp environment...")
        
        # bitnet.cppディレクトリを確認
        bitnet_dir = Path(__file__).parent / "bitnet"
        if not bitnet_dir.exists():
            raise FileNotFoundError(
                f"bitnet.cpp directory not found at {bitnet_dir}. "
                "Please clone the BitNet repository: git clone https://github.com/microsoft/BitNet.git bitnet"
            )
        
        # modelsディレクトリを作成
        models_dir = bitnet_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # GGUFモデルをダウンロード (初回実行時)
        self._download_gguf_model()
        
        print("bitnet.cpp environment is ready")
    
    def _download_gguf_model(self):
        """GGUFモデルをダウンロード"""
        from huggingface_hub import snapshot_download
        import warnings
        
        model_config = self.config['model']
        bitnet_config = self.config.get('bitnet_cpp', {})
        
        # モデルが既に存在する場合はスキップ
        model_path = bitnet_config.get('model_path', "")
        if model_path and os.path.exists(model_path):
            print(f"GGUF model already exists at {model_path}")
            return
        
        print(f"Downloading GGUF model: {model_config['huggingface_id']}")
        
        try:
            # HuggingFace Hubからモデルをダウンロード
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # bitnet.cpp用のGGUFモデルをダウンロード
                # 注: 実際のGGUFモデルはHuggingFace Hubにアップロードされる予定
                local_dir = Path(__file__).parent / "bitnet" / "models" / model_config['name']
                local_dir.mkdir(parents=True, exist_ok=True)
                
                # GGUFファイルを探す
                gguf_file = local_dir / model_config.get('gguf_model', "ggml-model-i2_s.gguf")
                
                if not gguf_file.exists():
                    # HuggingFace Hubからダウンロードを試みる
                    try:
                        snapshot_download(
                            repo_id=model_config['huggingface_id'],
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False,
                            ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.safetensors", "*.bin"]
                        )
                        print(f"Model downloaded to {local_dir}")
                    except Exception as e:
                        print(f"Failed to download from HuggingFace Hub: {e}")
                        print("Please manually download the GGUF model and place it in:")
                        print(f"  {local_dir / model_config.get('gguf_model', 'ggml-model-i2_s.gguf')}")
                        print("\nYou can find GGUF models at:")
                        print("  https://huggingface.co/models?search=bitnet")
                        
                # GGUFファイルの存在を確認
                if gguf_file.exists():
                    print(f"GGUF model found: {gguf_file}")
                    # configを更新
                    self.config['bitnet_cpp']['model_path'] = str(gguf_file)
                else:
                    print(f"Warning: GGUF model not found at {gguf_file}")
                    print("bitnet.cpp will use the default model path")
                    
        except ImportError:
            print("huggingface_hub is not installed. Skipping download.")
            print("Please install it: pip install huggingface-hub")
    
    def _run_bitnet_inference(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        bitnet.cppを使用して推論を実行
        
        Args:
            prompt: プロンプト
            max_tokens: 最大生成トークン数
            temperature: 温度
            
        Returns:
            生成されたテキスト
        """
        bitnet_dir = Path(__file__).parent / "bitnet"
        bitnet_config = self.config.get('bitnet_cpp', {})
        
        # model_pathを取得
        model_path = bitnet_config.get('model_path', "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf")
        
        # bitnet.cppのrun_inference.pyを呼び出す
        # あるいは、llama-cliを直接呼び出す
        
        # 方法1: run_inference.pyを使用
        run_inference_path = bitnet_dir / "run_inference.py"
        
        if run_inference_path.exists():
            # run_inference.pyを呼び出す
            import subprocess
            
            command = [
                sys.executable, str(run_inference_path),
                "-m", str(model_path),
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "-t", str(bitnet_config.get('threads', 2)),
                "-c", str(bitnet_config.get('ctx_size', 2048)),
                "-ngl", "0"  # GPUを使用しない
            ]
            
            print(f"Running command: {' '.join(command)}")
            
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # 出力から生成されたテキストを抽出
                    output = result.stdout
                    # プロンプトを除去
                    if prompt in output:
                        output = output[len(prompt):]
                    return output.strip()
                else:
                    print(f"Error: {result.stderr}")
                    raise RuntimeError(f"Failed to run bitnet.cpp inference: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                raise TimeoutError("bitnet.cpp inference timed out")
            except FileNotFoundError:
                print("run_inference.py not found. Trying to use llama-cli directly...")
                # 方法2: llama-cliを直接呼び出す
                return self._run_llama_cli(prompt, max_tokens, temperature)
        else:
            # 方法2: llama-cliを直接呼び出す
            return self._run_llama_cli(prompt, max_tokens, temperature)
    
    def _run_llama_cli(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        llama-cliを直接呼び出して推論を実行
        
        Args:
            prompt: プロンプト
            max_tokens: 最大生成トークン数
            temperature: 温度
            
        Returns:
            生成されたテキスト
        """
        bitnet_dir = Path(__file__).parent / "bitnet"
        bitnet_config = self.config.get('bitnet_cpp', {})
        model_path = bitnet_config.get('model_path', "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf")
        
        # llama-cliのパス
        if os.name == "nt":  # Windows
            llama_cli_path = bitnet_dir / "build" / "bin" / "Release" / "llama-cli.exe"
            if not llama_cli_path.exists():
                llama_cli_path = bitnet_dir / "build" / "bin" / "llama-cli"
        else:  # Linux/Mac
            llama_cli_path = bitnet_dir / "build" / "bin" / "llama-cli"
        
        if not llama_cli_path.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {llama_cli_path}. "
                "Please build bitnet.cpp first: cd bitnet && mkdir build && cd build && cmake .. && make"
            )
        
        command = [
            str(llama_cli_path),
            "-m", str(model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-t", str(bitnet_config.get('threads', 2)),
            "-c", str(bitnet_config.get('ctx_size', 2048)),
            "-ngl", "0"  # GPUを使用しない
        ]
        
        print(f"Running llama-cli: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                output = result.stdout
                if prompt in output:
                    output = output[len(prompt):]
                return output.strip()
            else:
                print(f"Error: {result.stderr}")
                raise RuntimeError(f"Failed to run llama-cli: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise TimeoutError("llama-cli timed out")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        テキストを生成
        
        Args:
            prompt: プロンプト
            max_new_tokens: 最大生成トークン数
            temperature: 温度
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング
            stop_tokens: 止めるトークン
            
        Returns:
            生成されたテキスト
        """
        # 設定をマージ
        inference_config = self.config['inference']
        max_new_tokens = max_new_tokens or inference_config['max_new_tokens']
        temperature = temperature or inference_config['temperature']
        
        # bitnet.cppを使用して推論
        return self._run_bitnet_inference(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        コードを生成
        
        Args:
            prompt: コード生成の要件
            language: プログラミング言語
            max_tokens: 最大生成トークン数
            
        Returns:
            生成されたコード
        """
        # コード生成用のシステムプロンプト
        codegen_config = self.config['codegen']
        system_prompt = codegen_config['system_prompt']
        
        # 言語特有のプロンプトを追加
        language_hints = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "java": "Java",
            "c++": "C++",
            "go": "Go",
            "rust": "Rust",
            "ruby": "Ruby",
            "php": "PHP",
            "swift": "Swift",
            "kotlin": "Kotlin",
        }
        
        lang = language_hints.get(language.lower(), "Python")
        
        # 完全なプロンプトを作成
        full_prompt = f"""[{lang} Code Generation]
{system_prompt.format(prompt=prompt)}
"""
        
        # 最大トークン数
        max_tokens = max_tokens or codegen_config['max_code_tokens']
        temperature = codegen_config['temperature']
        
        # コード生成
        code = self.generate(
            prompt=full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return code
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        model_config = self.config['model']
        bitnet_config = self.config.get('bitnet_cpp', {})
        
        return {
            "name": model_config['name'],
            "huggingface_id": model_config['huggingface_id'],
            "gguf_model": model_config.get('gguf_model'),
            "quantization": model_config['quantization'],
            "engine": model_config.get('engine', 'bitnet.cpp'),
            "device": model_config['device'],
            "model_path": bitnet_config.get('model_path'),
        }


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BitNet b1.58-2B-4T Code Generator (bitnet.cpp)")
    parser.add_argument("--prompt", type=str, default="Hello, world! in Python", 
                       help="Code generation prompt")
    parser.add_argument("--language", type=str, default="python",
                       help="Programming language")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    args = parser.parse_args()
    
    # コード生成器を初期化
    print("Initializing BitNet b1.58-2B-4T Code Generator with bitnet.cpp...")
    
    try:
        generator = BitNetCodeGenerator()
        
        # モデル情報を表示
        model_info = generator.get_model_info()
        print(f"\nLoaded Model: {model_info['name']}")
        print(f"HuggingFace ID: {model_info['huggingface_id']}")
        print(f"Engine: {model_info['engine']}")
        print(f"Quantization: {model_info['quantization'].get('bits', '1')}bit")
        print(f"Model Path: {model_info.get('model_path', 'Not set')}")
        print("-" * 50)
        
        print(f"Generating {args.language} code for: {args.prompt}")
        print("-" * 50)
        
        # コード生成
        code = generator.generate_code(
            prompt=args.prompt,
            language=args.language,
            max_tokens=args.max_tokens
        )
        
        print(code)
        print("-" * 50)
        
        # 出力ファイルに保存
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"Code saved to: {args.output}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use bitnet.cpp, you need to:")
        print("1. Clone the BitNet repository: git clone https://github.com/microsoft/BitNet.git bitnet")
        print("2. Build bitnet.cpp: cd bitnet && mkdir build && cd build && cmake .. && make")
        print("3. Download GGUF model: Place the GGUF file in bitnet/models/")
        print("\nFor more information, see: https://github.com/microsoft/BitNet")
        sys.exit(1)


if __name__ == "__main__":
    main()
