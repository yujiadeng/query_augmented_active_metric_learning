# -*- coding: utf-8 -*-
"""
Compare the feature weights learned by the proposed method and the MPCKmeans. Generate Figure 4 based on the pre-run results on urban land cover dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *

res = np.load('./real_data/urban_land_cover/max_nc_300/rep8.npz', allow_pickle=True)
A_hist = res['A_hist_penalize']
weights = np.diag(A_hist.item()[300])
idx = sorted(np.arange(len(weights)), key=lambda i: weights[i], reverse=True)

res2 = np.load('./real_data/urban_land_cover/max_nc_300/rep1_MPCKmeans.npz', allow_pickle=True)
A_hist2 = res2['A_hist']
weights2 = np.diag(A_hist2.item()[300])

# normalize weights
weights_n = weights / np.max(weights)
weights2_n = weights2 / np.max(weights2)

plt.figure(figsize=(16, 4))
plt.tight_layout()
plt.bar([str(i) for i in idx], weights_n[idx], color='blue', alpha=0.5, label='Proposed')
plt.bar([str(i) for i in idx], weights2_n[idx], color='orange', alpha=0.5, label='MPCKmeans')
plt.xlabel('Index')
plt.ylabel('Normalized feature weight')
plt.xticks([])
plt.legend(loc='upper right')
plt.savefig('./figure/Figure_4.png', bbox_inches='tight')

