"""Type stubs for codegen module"""

from typing import Optional, List, Dict, Any

class BitLLMCodeGenerator:
    """1bit LLMを使用したコード生成器"""
    
    def __init__(self, config_path: str = "config.yaml") -> None:
        """コード生成器を初期化"""
        ...
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """テキストを生成"""
        ...
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: Optional[int] = None
    ) -> str:
        """コードを生成"""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        ...
