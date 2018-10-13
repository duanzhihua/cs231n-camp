# -*- coding: utf-8 -*-
import numpy as np
np.prod([1.,2.])
np.prod([[1.,2.],[3.,4.]])
np.prod([[1.,2.],[3.,4.]], axis=1)
 
x = np.array([1, 2, 3], dtype=np.uint8)
np.prod(x).dtype == np.uint


x = np.array([1, 2, 3], dtype=np.int8)
np.prod(x).dtype == np.int



# =============================================================================
# np.prod([1.,2.])
# Out[18]: 2.0
# 
# np.prod([[1.,2.],[3.,4.]])
# Out[19]: 24.0
# 
# np.prod([[1.,2.],[3.,4.]], axis=1)
# Out[20]: array([ 2., 12.])
# 
# x = np.array([1, 2, 3], dtype=np.uint8)
# 
# np.prod(x).dtype == np.uint
# Out[22]: True
# 
# x = np.array([1, 2, 3], dtype=np.int8)
# np.prod(x).dtype == np.int
# 
# 
# Out[23]: True
# 
# =============================================================================
