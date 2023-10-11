import numpy as np

from FMP.MatlabCode.gettime import gettime
from FMP.My_Code.ReadTool import get_feature

filepath = 'Data/BCI_data.mat'
after_processing = get_feature(filepath)
my_array = np.array(after_processing, dtype=object)
print(type(my_array))
gettime(my_array, 3)

# gettime(after_processing, 3)
