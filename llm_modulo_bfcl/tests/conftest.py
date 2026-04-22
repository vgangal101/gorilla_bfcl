"""pytest configuration: add llm_modulo_bfcl/ to sys.path so bare imports work."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
