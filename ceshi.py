import torch
import numpy as np


prediction = torch.tensor([[[1,2,3,4,0.6],
                            [1,2,3,0,0.4],
                            [1,2,9,6,0.3],
                            [1,2,3,4,0.2],
                            [1,2,3,4,0.8],]])
# # print(prediction[:, :, 1:4].shape)
# a, b = torch.max(prediction[:, :, 1:4], 2)
# # print(a.shape, b.shape)
# image_pred = prediction[0, :, :]
# print(image_pred[4])
# non_zero_ind = torch.nonzero(image_pred[:, 3]).squeeze()
# # print(non_zero_ind)
# # print(non_zero_ind.shape)
# dd = prediction[0,:,:]
# print(dd[non_zero_ind].shape)
# conf_sort_index = torch.sort(image_pred[:, 4], descending=True)[1]
# print(conf_sort_index)
# print(image_pred[4,1])
# confidence = 0.4
# # print(prediction.shape)
# conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
# print(conf_mask,conf_mask.shape)
# prediction = prediction * conf_mask
# print(prediction)
# image_pred = prediction[0, :, :]
# print(image_pred[2:])

# im_dim_list = np.random.rand(3, 2)
# print(im_dim_list.shape)
# print(im_dim_list.repeat(1, 2))
print(torch.cuda.is_available())