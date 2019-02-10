#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Input
serie = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
# Solución
serie.value_counts()

