#!/usr/bin/env python3
"""
1bit LLM コード生成スクリプト

GitHub Actionsで動作する1bit量子化LLMを使用したコード生成
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
        # 最後の生成されたトークンを取得
        last_token = input_ids[0][-1]
        last_token_str = self.stop_tokens[0]  # 簡易実装
        return False  # 今後拡張


class BitLLMCodeGenerator:
    """1bit LLMを使用したコード生成器"""
    
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
    
    def _load_model(self):
        """モデルとトークナイザーを読み込む"""
        print("Loading 1bit LLM model...")
        
        model_config = self.config['model']
        
        # トークナイザーの読み込み
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['huggingface_id'],
            trust_remote_code=True
        )
        
        # モデルの読み込み (1bit量子化)
        print("Loading model with 1bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['huggingface_id'],
            load_in_1bit=model_config['quantization']['load_in_1bit'],
            device_map=model_config['device_map'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # GPUが利用可能な場合はGPUに移動
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        
        print(f"Model loaded successfully: {model_config['name']}")
        print(f"Device: {model_config['device']}")
        
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
        return {
            "name": self.config['model']['name'],
            "huggingface_id": self.config['model']['huggingface_id'],
            "quantization": self.config['model']['quantization'],
            "device": self.config['model']['device'],
            "parameters": "3B"  # 3Bパラメータ
        }


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="1bit LLM Code Generator")
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
    generator = BitLLMCodeGenerator()
    
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
    
    # モデル情報を表示
    model_info = generator.get_model_info()
    print(f"\nModel: {model_info['name']}")
    print(f"Device: {model_info['device']}")


if __name__ == "__main__":
    main()
