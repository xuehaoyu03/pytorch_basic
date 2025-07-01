import pandas as pd
import numpy as np

df = pd.DataFrame({
    '姓名': ['张三', '李四'],
    '年龄': [25, 30],
    '是否学生': [True, False]
})


df = np.array(df)
print(df.shape)