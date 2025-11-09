import os

PY = "pyeongchang.cc.gatech.edu"
VCVR = "vancouver.cc.gatech.edu"
HOSTNAME = os.uname().nodename
PROJECT_PARENT_DIR = "/home/asinha389/shared" if HOSTNAME == PY or HOSTNAME == VCVR else "/home/hice1/asinha389/scratch"
BALLU_ASSETS_DIR = os.path.join(PROJECT_PARENT_DIR, "BALLU_Project", "ballu_isclb_extension", "source", "ballu_isaac_extension", "ballu_isaac_extension", "ballu_assets")
NEXT_LAB_DATE = "11.11.2025"