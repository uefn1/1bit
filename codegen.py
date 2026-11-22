#!/usr/bin/env python3
"""
BitNet 1bit LLM コード生成スクリプト

GitHub Actionsで動作するBitNet 1bit量子化LLMを使用したコード生成
"""

import torch
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)


class CodeStoppingCriteria(StoppingCriteria):
    """コード生成用のカスタム停止基準"""
    
    def __init__(self, stop_tokens: List[str]):
        self.stop_tokens = stop_tokens
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return False


class BitLLMCodeGenerator:
    """BitNet 1bit / 4bit LLMを使用したコード生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        コード生成器を初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _is_bitnet_model(self, model_id: str) -> bool:
        """BitNetモデルかどうかを判定"""
        return "bitnet" in model_id.lower()
    
    def _load_model(self):
        """モデルとトークナイザーを読み込む"""
        print("Loading BitNet 1bit LLM model...")
        
        model_config = self.config['model']
        quant_config = model_config['quantization']
        
        # 設定ファイルからフォールバックモデルを取得
        fallback_models = self.config.get('fallback_models', [
            model_config['huggingface_id'],
            "microsoft/BitNet-3B-1bit",
            "microsoft/BitNet-1.5B-1bit",
            "TinyLlama/TinyLlama-1.1B",
            "Qwen/Qwen2-0.5B-Instruct",
        ])
        
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
                
                # 量子化設定を決定
                quantization_kwargs = {}
                
                # BitNetモデルの場合は1bit量子化
                if self._is_bitnet_model(model_id) and quant_config.get('load_in_1bit', False):
                    print(f"Loading {model_id} with 1bit quantization...")
                    quantization_kwargs = {
                        'load_in_1bit': True,
                    }
                # BitNetでない場合は4bit量子化
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
            # 量子化なしで試行 (FP16/INT8)
            for model_id in fallback_models:
                try:
                    print(f"Trying to load {model_id} without quantization...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map=model_config['device_map'],
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True
                    )
                    loaded_model_id = model_id
                    break
                except Exception as e:
                    print(f"Failed to load {model_id} without quantization: {e}")
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
        top_k = top_k or inference_config['top_k']
        top_p = top_p or inference_config['top_p']
        
        # 生成設定
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=inference_config['do_sample'],
            repetition_penalty=inference_config['repetition_penalty']
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
            temperature=temperature,
            stop_tokens=codegen_config['stop_tokens']
        )
        
        return code
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        model_info = {
            "name": self.config['model']['name'],
            "huggingface_id": self.config['model']['huggingface_id'],
            "quantization": self.config['model']['quantization'],
            "device": self.config['model']['device'],
        }
        
        # BitNetモデルかどうかを追加
        model_id = self.config['model']['huggingface_id']
        model_info['is_bitnet'] = self._is_bitnet_model(model_id)
        model_info['quantization_bits'] = (
            1 if model_info['is_bitnet'] and model_info['quantization'].get('load_in_1bit') else
            4 if model_info['quantization'].get('load_in_4bit') else
            8 if model_info['quantization'].get('load_in_8bit') else
            16
        )
        
        return model_info


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BitNet 1bit LLM Code Generator")
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
    print("Initializing BitNet 1bit LLM Code Generator...")
    generator = BitLLMCodeGenerator()
    
    # モデル情報を表示
    model_info = generator.get_model_info()
    print(f"\nLoaded Model: {model_info['name']}")
    print(f"HuggingFace ID: {model_info['huggingface_id']}")
    print(f"Quantization: {model_info['quantization_bits']}bit")
    print(f"Is BitNet: {model_info['is_bitnet']}")
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


if __name__ == "__main__":
    main()
