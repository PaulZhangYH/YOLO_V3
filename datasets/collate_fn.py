import numpy as np

"""修改collate_fn，以适应边界框数量不同的targets"""
def collate_fn(batch):
    img, targets = zip(*batch) # transposed
    maxLen = 0
    for i in range(len(targets)):
        maxLen = max(maxLen, len(targets[i]))
    padding_element = np.array([0., 0., 0., 0., 0.])
    for target in targets:
        num_padding = maxLen - len(target)
        for i in range(num_padding):
            target = np.append(target, padding_element)
    return np.array(img), np.array(targets)