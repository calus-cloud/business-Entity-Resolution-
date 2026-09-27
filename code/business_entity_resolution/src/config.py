"""Default paths (override with --data-dir / --work-dir, or env vars)."""
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEFAULT_DATA_DIR = os.environ.get(
    "ER_DATA_DIR", r"E:\6ab10eb3b23ba_student_resource\student_resource")
DEFAULT_WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(_ROOT, "work"))
DEFAULT_OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(_ROOT, "output"))
