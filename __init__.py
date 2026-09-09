"""MiniMax H3 V100 optimization node."""
from .h3_optimize import H3V100Optimize
__version__ = "2.0.0"
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {"H3V100Optimize": H3V100Optimize}
NODE_DISPLAY_NAME_MAPPINGS = {"H3V100Optimize": "H3 V100 Optimize"}
