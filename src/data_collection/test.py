#!/usr/bin/env python3
import os
# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt
import copy
import multiprocessing
import time
from tqdm import tqdm
# from collections import defaultdict
import h5py
# import cv2
import numpy as np

with h5py.File('/home/forceyqj/code_wb/flexiv/src/data/flexiv/teleop_recordings/2025-10-09/trajectory_20251009_110213.h5', 'r') as h5f: 
        cam1= h5f['cam1'][:] 
        camera_timestamps= h5f['camera_timestamps'][:] 
        action_timestamps = h5f['action_timestamps'][:] 
        # print(f"{camera_timestamps[0]}")
        # print(f"{camera_timestamps[700]}")
        print(f"[]: {camera_timestamps[0]:.15f}")