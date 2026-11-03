"""TraceFormer S1：论文模型与八组可重训消融。"""

from .config import TraceFormerConfig, load_config
from .variants import VARIANTS, build_model

__all__ = ["TraceFormerConfig", "load_config", "VARIANTS", "build_model"]
