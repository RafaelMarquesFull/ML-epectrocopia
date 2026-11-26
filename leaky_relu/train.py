"""Experimento leaky_relu: toda a lógica está em common/core.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from core import main  # noqa: E402

main("leaky_relu")
