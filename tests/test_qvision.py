import unittest
import numpy as np
import sys
import os
from qvision.gerch_sax import load_train_test_images, load_images_from_directory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qvision import QVision

class TestQVision(unittest.TestCase):
    def setUp(self):
        factor = 35
        lr_weights = 0.01/factor
        lr_bias = 0.001/factor
        self.model = QVision(input_shape=(28, 28), num_epochs=1200, lr_weights=lr_weights, lr_bias=lr_bias, num_shots=-1)

    def test_initialize_parameters(self):
        self.model.initialize_parameters()
        self.assertIsNotNone(self.model.weights, "Weights should be initialized")
        self.assertIsNotNone(self.model.bias, "Bias should be initialized")
        self.assertEqual(self.model.weights.shape, self.model.input_shape, "Weights shape should match input shape")

    def test_preprocess_data(self):
        # Create dummy data
        train_imgs = np.random.rand(10, 32, 32, 3)
        train_labels = np.random.randint(0, 2, 10)
        test_imgs = np.random.rand(5, 32, 32, 3)
        test_labels = np.random.randint(0, 2, 5)

        train_imgs, train_labels, test_imgs, test_labels = self.model.preprocess_data(train_imgs, train_labels, test_imgs, test_labels)

        # Check that images are converted to float
        self.assertEqual(train_imgs.dtype, np.float64, "Train images should be float64")
        self.assertEqual(test_imgs.dtype, np.float64, "Test images should be float64")

        # Check that images are normalized and amplitude calculated
        self.assertTrue(np.all(train_imgs >= 0) and np.all(train_imgs <= 1), "Train images should be normalized")
        self.assertTrue(np.all(test_imgs >= 0) and np.all(test_imgs <= 1), "Test images should be normalized")

    def test_train(self):
        optimizers = ['gd']
        results = {}

        # Create dummy data
        train_imgs = np.random.rand(10, 32, 32, 3)
        train_labels = np.random.randint(0, 2, 10)
        test_imgs = np.random.rand(5, 32, 32, 3)
        test_labels = np.random.randint(0, 2, 5)

        # Define the folders where the images are stored
        train_images_folder = '../qvision/training_images'
        test_images_folder = '../qvision/test_images'

        # Load the data
        (train_source_images, train_modulated_images, train_labels), \
        (test_source_images, test_modulated_images, test_labels) = load_train_test_images(train_images_folder, test_images_folder)

        # Shuffle the training set (source images, modulated images, and labels)
        # train_source_images, train_modulated_images, train_labels = shuffle_dataset(train_source_images, train_modulated_images, train_labels)

        # train_imgs, train_labels, test_imgs, test_labels = self.model.preprocess_data(train_imgs, train_labels,
        #                                                                               test_imgs, test_labels)

        self.model.initialize_parameters()

        for optimizer in optimizers:
            print(f'Training with {optimizer} optimizer...')
            weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = self.model.train(
                optimizer, None, None, None, None, train_source_images[:4000], train_modulated_images[:4000],
                train_labels[:4000], test_source_images[:1000], test_modulated_images[:1000], test_labels[:1000], phase_modulation=True)
            results[optimizer] = {
                'weights': weights,
                'bias': bias,
                'loss_history': loss_history,
                'test_loss_history': test_loss_history,
                'accuracy_history': accuracy_history,
                'test_accuracy_history': test_accuracy_history,
            }

        # Evaluate and compare results
        for optimizer, result in results.items():
            print(f'Optimizer: {optimizer}')
            print(f'Final training loss: {result["loss_history"][-1]}')
            print(f'Final test loss: {result["test_loss_history"][-1]}')
            print(f'Final training accuracy: {result["accuracy_history"][-1]}')
            print(f'Final test accuracy: {result["test_accuracy_history"][-1]}')
            print('---')

        # Determine best optimizer based on test accuracy or other metrics
        best_optimizer = max(results, key=lambda x: results[x]['test_accuracy_history'][-1])
        print(f'Best optimizer: {best_optimizer}')

        return results, best_optimizer


if __name__ == '__main__':
    unittest.main()
