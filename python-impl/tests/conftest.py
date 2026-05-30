# 配置 pytest 导入路径，使测试可直接加载 src 包。

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
