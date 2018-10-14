# -*- coding: utf-8 -*-
import numpy as np
x = np.array([[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]])
rows = np.array([[0, 0],
                  [3, 3]], dtype=np.intp)
columns = np.array([[0, 2],
                     [0, 2]], dtype=np.intp)
x[rows, columns]

rows = np.array([0, 3], dtype=np.intp)
columns = np.array([0, 2], dtype=np.intp)
rows[:, np.newaxis]
x[rows[:, np.newaxis], columns]