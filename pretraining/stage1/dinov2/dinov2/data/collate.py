# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    # print('data0:',samples_list[0].keys())#data0: dict_keys(['image', 'birads_label', 'density_label'])
    
    images = [s['image'] for s in samples_list]  
    birads_labels = [s['birads_label'] for s in samples_list]  
    density_labels = [s['density_label'] for s in samples_list] 

    n_global_crops = len(images[0]["global_crops"])
    n_local_crops = len(images[0]["local_crops"])

    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in images])

    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in images])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    birads_labels = [label for label in birads_labels for _ in range(n_global_crops)]
    density_labels = [label for label in density_labels for _ in range(n_global_crops)]

    birads_labels_tensor = torch.tensor(birads_labels)   
    density_labels_tensor = torch.tensor(density_labels)  

    return {
        "birads_label": birads_labels_tensor,
        "density_label": density_labels_tensor,
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }










# # Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# # This source code is licensed under the Apache License, Version 2.0
# # found in the LICENSE file in the root directory of this source tree.

# import torch
# import random


# def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
#     # dtype = torch.half  # TODO: Remove
#     print()
#     n_global_crops = len(samples_list[0]["global_crops"])  
#     n_local_crops = len(samples_list[0]["local_crops"])  
#     # 合并 global crops
#     collated_global_crops = torch.stack(
#         [sample["global_crops"][i] for i in range(n_global_crops) for sample in samples_list]
#     )
#     # 合并 local crops
#     collated_local_crops = torch.stack(
#         [sample["local_crops"][i] for i in range(n_local_crops) for sample in samples_list]
#     )

#     # n_global_crops = len(samples_list[0][0]["global_crops"])
#     # n_local_crops = len(samples_list[0][0]["local_crops"])

#     # collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

#     # collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

#     B = len(collated_global_crops)
#     N = n_tokens
#     n_samples_masked = int(B * mask_probability)
#     probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
#     upperbound = 0
#     masks_list = []
#     for i in range(0, n_samples_masked):
#         prob_min = probs[i]
#         prob_max = probs[i + 1]
#         masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
#         upperbound += int(N * prob_max)
#     for i in range(n_samples_masked, B):
#         masks_list.append(torch.BoolTensor(mask_generator(0)))

#     random.shuffle(masks_list)

#     collated_masks = torch.stack(masks_list).flatten(1)
#     mask_indices_list = collated_masks.flatten().nonzero().flatten()

#     masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
#     birads_label = samples_list[0]["birads_label"]
#     density_label = samples_list[0]["density_label"]

#     return {
#         "birads_label": torch.stack([birads_label,birads_label]).to(dtype),
#         "density_label": torch.stack([density_label,density_label]).to(dtype),
#         "collated_global_crops": collated_global_crops.to(dtype),
#         "collated_local_crops": collated_local_crops.to(dtype),
#         "collated_masks": collated_masks,
#         "mask_indices_list": mask_indices_list,
#         "masks_weight": masks_weight,
#         "upperbound": upperbound,
#         "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
#     }
