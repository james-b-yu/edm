import torch

coords_before = torch.tensor([[[-0.2356, -0.5774,  0.6368],
                               [ 1.7486, -1.5261,  1.7175],
                               [ 2.4272, -2.3563,  2.1409],
                               [ 1.4315,  0.9962,  1.2512],
                               [ 0.3985,  1.7100,  0.3501],
                               [-1.4366,  0.8426, -0.9612],
                               [ 0.9716, -0.4623,  1.2017],
                               [-2.6625,  0.1475, -1.0645],
                               [-0.8276,  0.7634,  0.3130]]], device='mps:0')

coords_after = torch.tensor([[[-0.4373, -0.5260,  0.0162],
                              [ 1.5469, -1.4747,  1.0969],
                              [ 2.2255, -2.3050,  1.5203],
                              [ 1.2298,  1.0476,  0.6306],
                              [ 0.1968,  1.7613, -0.2706],
                              [-1.6383,  0.8939, -1.5818],
                              [ 0.7699, -0.4110,  0.5811],
                              [-2.8641,  0.1989, -1.6851],
                              [-1.0293,  0.8148, -0.3076]]], device='mps:0')

# Compute the mean of the coordinates
mean_before = coords_before.mean(dim=1)
mean_after = coords_after.mean(dim=1)

print("Mean before normalization:", mean_before)
print("Mean after normalization:", mean_after)


import torch

coords_before = torch.tensor([[[-0.2356, -0.5774,  0.6368],
                               [ 1.7486, -1.5261,  1.7175],
                               [ 2.4272, -2.3563,  2.1409],
                               [ 1.4315,  0.9962,  1.2512],
                               [ 0.3985,  1.7100,  0.3501],
                               [-1.4366,  0.8426, -0.9612],
                               [ 0.9716, -0.4623,  1.2017],
                               [-2.6625,  0.1475, -1.0645],
                               [-0.8276,  0.7634,  0.3130]]], device='mps:0')

coords_after = torch.tensor([[[-0.4373, -0.5260,  0.0162],
                              [ 1.5469, -1.4747,  1.0969],
                              [ 2.2255, -2.3050,  1.5203],
                              [ 1.2298,  1.0476,  0.6306],
                              [ 0.1968,  1.7613, -0.2706],
                              [-1.6383,  0.8939, -1.5818],
                              [ 0.7699, -0.4110,  0.5811],
                              [-2.8641,  0.1989, -1.6851],
                              [-1.0293,  0.8148, -0.3076]]], device='mps:0')

# Compute mean and standard deviation
mean_before = coords_before.mean(dim=1)
std_before = coords_before.std(dim=1)

mean_after = coords_after.mean(dim=1)
std_after = coords_after.std(dim=1)

print("Mean before normalization:", mean_before)
print("Standard deviation before normalization:", std_before)

print("Mean after normalization:", mean_after)
print("Standard deviation after normalization:", std_after)
