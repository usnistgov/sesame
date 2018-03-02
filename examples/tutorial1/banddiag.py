import sesame
import numpy as np

sys, result = sesame.load_sim('1dhomo_IV_0.gzip')
az = sesame.Analyzer(sys,result)
p1 = (0,0)
p2 = (3e-4,0)
az.band_diagram((p1,p2))
