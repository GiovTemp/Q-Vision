import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# def gerch_sax(I, F, N_tot=2000):
#     n, m = F.shape
#     D = F
#     trend = np.zeros(N_tot)
#
#     for i in range(N_tot):
#         A = np.fft.ifft2(D) * np.sqrt(n * m)
#         B = I * np.exp(1j * np.angle(A))
#         C = np.fft.fft2(B) / np.sqrt(n * m)
#         D = F * np.exp(1j * np.angle(C))
#         trend[i] = np.linalg.norm(np.abs(C) - np.abs(F))
#
#     return B, C
#
# def load_images_from_directory(folder_path):
#     images = []
#     labels = []
#
#     # Iterate over the files in the folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.npy'):
#             # Load the image (amplitude values) from the .npy file
#             image = np.load(os.path.join(folder_path, filename))
#
#             # Extract the label from the filename
#             label = float(filename.split('_')[-1].split('.')[0])
#
#             images.append(image)
#             labels.append(label)
#
#     # Convert to NumPy arrays for compatibility with deep learning libraries
#     images = np.array(images)
#     labels = np.array(labels)
#
#     return images, labels
#
# def load_train_test_images(train_folder, test_folder):
#     # Load training source and modulated images
#     train_source_folder = f'{train_folder}/source_images'
#     train_modulated_folder = f'{train_folder}/modulated_images'
#
#     # Load test source and modulated images
#     test_source_folder = f'{test_folder}/source_images'
#     test_modulated_folder = f'{test_folder}/modulated_images'
#
#     # Load training images
#     train_source_images, train_labels = load_images_from_directory(train_source_folder)
#     train_modulated_images, _ = load_images_from_directory(train_modulated_folder)
#
#     # Load testing images
#     test_source_images, test_labels = load_images_from_directory(test_source_folder)
#     test_modulated_images, _ = load_images_from_directory(test_modulated_folder)
#
#     return (train_source_images, train_modulated_images, train_labels), \
#            (test_source_images, test_modulated_images, test_labels)
#
# # m = np.zeros((32, 32))
# #
# # F = np.ones((32, 32))
# # F = np.sqrt(F / np.sum(F))
# #
# # m[14:18, 14:18] = np.ones((4, 4))
# #
# # m = np.sqrt(m / np.sum(m))
# #
# # # print(np.linalg.norm(m))
# #
# # _, C = gerch_sax(m, F)
#
# # Define the folders where the images are stored
# train_images_folder = 'training_images'
# test_images_folder = 'test_images'
#
# # Load the data
# (train_source_images, train_modulated_images, train_labels), \
# (test_source_images, test_modulated_images, test_labels) = load_train_test_images(train_images_folder, test_images_folder)
#
# k_zero = 0
# k_one = 0
#
# modulated_images_zero = []
# modulated_images_one = []
#
#
# for i in range(2000):
#     if train_labels[i] == 0:
#         k_zero = k_zero + 1
#         modulated_images_zero.append(train_modulated_images[i, :, :])
#
#     if k_zero == 50:
#         break
#
#
# for i in range(2000):
#     if train_labels[i] == 1:
#         k_one = k_one + 1
#         modulated_images_one.append(train_modulated_images[i, :, :])
#
#     if k_one == 50:
#         break
#
# modulated_images_zero = np.array(modulated_images_zero, dtype=np.complex128)
# modulated_images_one = np.array(modulated_images_one, dtype=np.complex128)
#
# img_norm = modulated_images_zero[0, :, :]
#
# trace_0 = []
# trace_1 = []
#
# for i in range(1, 50):
#     img_transp_zero = np.transpose(modulated_images_zero[i, :, :])
#     img_transp_zero = np.conj(img_transp_zero)
#     img_transp_one = np.transpose(modulated_images_one[i, :, :])
#     img_transp_one = np.conj(img_transp_one)
#     prod_zero = np.dot(img_norm, img_transp_zero)
#     prod_one = np.dot(img_norm, img_transp_one)
#
#     trace0 = np.trace(prod_zero)
#     trace1 = np.trace(prod_one)
#
#     trac_abs_sq_0 = np.square(np.abs(trace0))
#     trac_abs_sq_1 = np.square(np.abs(trace1))
#
#     trace_0.append(trac_abs_sq_0)
#     trace_1.append(trac_abs_sq_1)
#
# figure, axis = plt.subplots(2, 2)
#
# # For Sine Function
# axis[0, 0].hist(trace_0)
# axis[0, 0].set_title("Trace 0")
#
# # For Cosine Function
# axis[0, 1].hist(trace_1)
# axis[0, 1].set_title("Trace 1")
#
# plt.show()
#
# # plt.hist(trace_0)
# # plt.show()
# #
# # plt.hist(trace_1)
# # plt.show()
#
#
#
#

m = np.ones((2, 3))
print(np.size(m))
