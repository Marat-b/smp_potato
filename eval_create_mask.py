import numpy as np

#https://chowdera.com/2022/04/202204110503041424.html
CLASSES = ['unlabelled', 'potato_strong', 'potato_sick', 'stone'
           ]
msk =np.asarray([
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2]
])

# msk = np.asarray([
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1]
# ])

classes = CLASSES
self_class_values = [CLASSES.index(cls.lower()) for cls in classes]
print(f'self_class_values={self_class_values}')
# for v in self_class_values:
#     print(f'v={v}')
# print((msk == 1))
masks = [(msk == v) for v in self_class_values]
print(f'masks={masks}')
mask = np.stack(masks, axis=-1).astype('float')
print(f'mask.shape={mask.shape}')
tmask = mask.transpose((2, 1, 0))
print(f'tmask.shape={tmask.shape}')
print(f'tmask={tmask}')
