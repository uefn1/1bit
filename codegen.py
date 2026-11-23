#!/usr/bin/env python3
"""
BitNet b1.58-2B-4T コード生成スクリプト

HuggingFace Transformers (1bit量子化) または bitnet.cpp (C++推論エンジン) を使用
"""

import os
import sys
import subprocess
import yaml
import tempfile
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any

# HuggingFace Transformersを使用するかどうか
USE_TRANSFORMERS = True

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        StoppingCriteria,
        StoppingCriteriaList
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    USE_TRANSFORMERS = False


class CodeStoppingCriteria(StoppingCriteria):
    """コード生成用のカスタム停止基準"""
    
    def __init__(self, stop_tokens: List[str]):
        self.stop_tokens = stop_tokens
        
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return False


class BitNetCodeGenerator:
    """BitNet b1.58-2B-4Tを使用したコード生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        コード生成器を初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.engine = None
        
        # 推論エンジンを決定
        engine = self.config['model'].get('engine', 'transformers')
        
        if engine == 'transformers' or not self._can_use_bitnet_cpp():
            self._load_transformers_model()
            self.engine = 'transformers'
        else:
            self._load_bitnet_cpp_model()
            self.engine = 'bitnet.cpp'
        
        print(f"Using engine: {self.engine}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _can_use_bitnet_cpp(self) -> bool:
        """bitnet.cppが使用可能かを確認"""
        bitnet_dir = Path(__file__).parent / "bitnet"
        if not bitnet_dir.exists():
            return False
        
        # llama-cliが存在するかを確認
        if os.name == "nt":
            llama_cli = bitnet_dir / "build" / "bin" / "Release" / "llama-cli.exe"
            if not llama_cli.exists():
                llama_cli = bitnet_dir / "build" / "bin" / "llama-cli"
        else:
            llama_cli = bitnet_dir / "build" / "bin" / "llama-cli"
        
        return llama_cli.exists()
    
    def _is_bitnet_model(self, model_id: str) -> bool:
        """BitNetモデルかどうかを判定"""
        return "bitnet" in model_id.lower()

    def _download_model_from_url(self, url: str, target_dir: Path) -> Path:
        """Download model.safetensors and all required files from URL and return the directory path
        
        Supports:
        - Individual files from GitHub releases (e.g., model.safetensors)
        - Directory URLs containing all model files
        - URL-encoded paths (e.g., microsoft%2Fbitnet-b1.58-2B-4T)
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode URL-encoded characters (e.g., %2F -> /)
        from urllib.parse import unquote
        url = unquote(url)
        
        # Check if URL points to a zip file
        if url.lower().endswith(".zip"):
            print(f"Downloading and extracting zip file from {url}...")
            import zipfile
            import io
            try:
                with urllib.request.urlopen(url) as response:
                    with zipfile.ZipFile(io.BytesIO(response.read())) as zip_ref:
                        zip_ref.extractall(target_dir)
                print(f"Zip file extracted to {target_dir}")
                return target_dir
            except Exception as e:
                raise RuntimeError(f"Failed to download and extract zip file: {e}")
        
        # Extract base URL (directory containing the files)
        # URL format: https://github.com/uefn1/1bit/releases/download/microsoft%2Fbitnet-b1.58-2B-4T/model.safetensors
        # We want: https://github.com/uefn1/1bit/releases/download/microsoft%2Fbitnet-b1.58-2B-4T/
        if url.endswith("model.safetensors"):
            base_url = url[:-len("model.safetensors")]
        else:
            # If it's a directory URL, use it as is
            base_url = url if url.endswith("/") else url + "/"
        
        print(f"Downloading model files from: {base_url}")
        
        # List of required files
        required_files = [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        
        # Download all required files
        for file_name in required_files:
            file_url = base_url + file_name
            file_path = target_dir / file_name
            print(f"  Downloading {file_name}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
                print(f"  Saved to {file_path}")
            except Exception as e:
                print(f"  Warning: Could not download {file_name} from {file_url}: {e}")
                # For tokenizer files, try to get from official model
                if file_name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                    print(f"  Attempting to download from official model...")
                    try:
                        from transformers import AutoTokenizer
                        hf_model_id = "microsoft/bitnet-b1.58-2B-4T"
                        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                        tokenizer.save_pretrained(target_dir)
                        print(f"  Tokenizer files saved from {hf_model_id}")
                        break  # Only need to do this once
                    except Exception as e2:
                        print(f"  Error: Could not download tokenizer: {e2}")
        
        return target_dir
    def _load_transformers_model(self):
        """HuggingFace Transformersを使用してモデルを読み込む"""
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is not installed. "
                "Please install it: pip install transformers torch bitsandbytes"
            )
        
        print("Loading BitNet model with HuggingFace Transformers...")
        
        model_config = self.config['model']
        quant_config = model_config['quantization']
        
        # Check if custom model URL is provided
        custom_model_url = model_config.get("model_url")
        custom_local_path = model_config.get("local_model_path")
        
        if custom_model_url:
            # Download from custom URL
            cache_dir = Path.home() / ".cache" / "bitnet_models"
            model_dir = cache_dir / "custom_model"
            print(f"Downloading model from custom URL: {custom_model_url}")
            model_path = self._download_model_from_url(custom_model_url, model_dir)
            fallback_models = [str(model_path)]
        elif custom_local_path and os.path.exists(custom_local_path):
            # Use local path
            print(f"Using local model from: {custom_local_path}")
            fallback_models = [custom_local_path]
        else:
            # Use HuggingFace Hub fallback models
            fallback_models = [
                model_config["huggingface_id"],
                "microsoft/bitnet-b1.58-2B-4T",
                "microsoft/bitnet-b1.58-2B-4T-bf16",
            ]
        loaded_model_id = None
        
        # モデルを読み込む (フォールバックあり)
        for model_id in fallback_models:
            try:
                print(f"Trying to load model: {model_id}")
                
                # トークナイザーの読み込み
                print(f"Loading tokenizer for {model_id}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                
                # Quantization settings
                quantization_kwargs = {}
                
                # BitNet models are already pre-quantized (1.58-bit), no quantization kwargs needed
                if self._is_bitnet_model(model_id):
                    pass  # BitNet models use their own quantization
                elif quant_config.get('load_in_4bit', False):
                    print(f"Loading {model_id} with 4bit quantization...")
                    quantization_kwargs = {
                        'load_in_4bit': True,
                        'bnb_4bit_compute_dtype': torch.float16,
                        'bnb_4bit_quant_type': 'nf4',
                        'bnb_4bit_use_double_quant': True,
                    }
                elif quant_config.get('load_in_8bit', False):
                    print(f"Loading {model_id} with 8bit quantization...")
                    quantization_kwargs = {'load_in_8bit': True}
                
                # モデルの読み込み
                print(f"Loading model: {model_id}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=model_config['device_map'],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    **quantization_kwargs
                )
                
                loaded_model_id = model_id
                break
                
            except Exception as e:
                print(f"Failed to load {model_id}: {e}")
                continue
        
        if loaded_model_id is None:
            raise RuntimeError(
                "All fallback models failed to load. "
                "Please check your internet connection and HuggingFace Hub access. "
                "You may need to run 'huggingface-cli login' to authenticate. "
                f"Tried models: {fallback_models}"
            )
        
        # GPUが利用可能な場合はGPUに移動
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        
        print(f"Model loaded successfully: {loaded_model_id}")
        print(f"Device: {model_config['device']}")
        
        # 実際にロードしたモデルIDを更新
        self.config['model']['huggingface_id'] = loaded_model_id
        self.config['model']['name'] = loaded_model_id.split('/')[-1]
        
    def _load_bitnet_cpp_model(self):
        """bitnet.cppを使用してモデルを読み込む"""
        print("Loading BitNet model with bitnet.cpp...")
        
        bitnet_dir = Path(__file__).parent / "bitnet"
        if not bitnet_dir.exists():
            raise FileNotFoundError(
                f"bitnet.cpp directory not found at {bitnet_dir}. "
                "Please clone the BitNet repository: git clone https://github.com/microsoft/BitNet.git bitnet"
            )
        
        # modelsディレクトリを作成
        models_dir = bitnet_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # GGUFモデルを確認
        model_config = self.config['model']
        bitnet_config = self.config.get('bitnet_cpp', {})
        model_path = bitnet_config.get('model_path')
        
        if model_path and not os.path.exists(model_path):
            print(f"GGUF model not found at {model_path}")
            print("Please download the GGUF model and place it in the correct location")
            model_path = None
        
        # llama-cliの存在を確認
        if os.name == "nt":
            llama_cli = bitnet_dir / "build" / "bin" / "Release" / "llama-cli.exe"
            if not llama_cli.exists():
                llama_cli = bitnet_dir / "build" / "bin" / "llama-cli"
        else:
            llama_cli = bitnet_dir / "build" / "bin" / "llama-cli"
        
        if not llama_cli.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {llama_cli}. "
                "Please build bitnet.cpp first: cd bitnet && mkdir build && cd build && cmake .. && make"
            )
        
        self.llama_cli_path = str(llama_cli)
        self.model_path = model_path or bitnet_config.get('model_path', 
            "bitnet/models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf")
        self.threads = bitnet_config.get('threads', 2)
        self.ctx_size = bitnet_config.get('ctx_size', 2048)
        
        print(f"bitnet.cpp loaded successfully")
        print(f"llama-cli: {self.llama_cli_path}")
        print(f"Model path: {self.model_path}")
        
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
        command = [
            self.llama_cli_path,
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
            "-ngl", "0"  # GPUを使用しない
        ]
        
        print(f"Running: {' '.join(command)}")
        
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
        
        # エンジンに応じて推論を実行
        if self.engine == 'transformers':
            return self._generate_with_transformers(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=stop_tokens
            )
        else:  # bitnet.cpp
            return self._run_bitnet_inference(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
    
    def _generate_with_transformers(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """HuggingFace Transformersを使用してテキストを生成"""
        # 生成設定
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=self.config['inference'].get('repetition_penalty', 1.1)
        )
        
        # 止める基準
        stopping_criteria = StoppingCriteriaList()
        if stop_tokens:
            stopping_criteria.append(CodeStoppingCriteria(stop_tokens))
        
        # トークナイズ
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # デバイスに移動
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria
            )
        
        # デコード
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # プロンプトを除去
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()
    
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
        
        info = {
            "name": model_config['name'],
            "huggingface_id": model_config['huggingface_id'],
            "quantization": model_config['quantization'],
            "engine": self.engine,
            "device": model_config['device'],
        }
        
        if self.engine == 'bitnet.cpp':
            info['llama_cli_path'] = self.llama_cli_path
            info['model_path'] = self.model_path
        
        return info


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BitNet b1.58-2B-4T Code Generator (Transformers or bitnet.cpp)"
    )
    parser.add_argument("--prompt", type=str, default="Hello, world! in Python", 
                       help="Code generation prompt")
    parser.add_argument("--language", type=str, default="python",
                       help="Programming language")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    parser.add_argument("--model-url", type=str, default=None,
                       help="Custom model URL (e.g., GitHub releases)")
    parser.add_argument("--local-model", type=str, default=None,
                       help="Local model directory path")
    args = parser.parse_args()
    
    # コード生成器を初期化
    print("Initializing BitNet b1.58-2B-4T Code Generator...")
    
    try:
        # Handle custom model paths from CLI
        if args.model_url or args.local_model:
            # Update config with CLI arguments
            import yaml
            with open("config.yaml", "r", encoding="utf-8") as cf:
                config = yaml.safe_load(cf)
            if args.model_url:
                config["model"]["model_url"] = args.model_url
            if args.local_model:
                config["model"]["local_model_path"] = args.local_model
            # Save updated config to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
                yaml.dump(config, tf)
                temp_config = tf.name
            generator = BitNetCodeGenerator(config_path=temp_config)
            # Clean up temp file
            import os
            os.unlink(temp_config)
        
        # モデル情報を表示
        model_info = generator.get_model_info()
        print(f"\nLoaded Model: {model_info['name']}")
        print(f"HuggingFace ID: {model_info['huggingface_id']}")
        print(f"Engine: {model_info['engine']}")
        print(f"Quantization: {model_info['quantization'].get('bits', '1')}bit")
        print(f"Device: {model_info['device']}")
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
        print("\nTo use this code generator, you have two options:")
        print("\n🔹 Option 1: Use HuggingFace Transformers (Recommended)")
        print("   pip install transformers torch bitsandbytes")
        print("   # No build required, works out of the box")
        print("\n🔹 Option 2: Use bitnet.cpp (C++ engine)")
        print("   1. Clone: git clone https://github.com/microsoft/BitNet.git bitnet")
        print("   2. Build: cd bitnet && mkdir build && cd build && cmake .. && make")
        print("   3. Download GGUF model and place it in bitnet/models/")
        print("\nFor more information, see: https://github.com/microsoft/BitNet")
        sys.exit(1)


if __name__ == "__main__":
    main()
