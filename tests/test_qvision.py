import unittest
import numpy as np
from qvision import QVision

class TestQVision(unittest.TestCase):
    def setUp(self):
        self.model = QVision(num_epochs=10, lr_weights=0.075, lr_bias=0.005, num_shots=-1)

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
        # Create dummy data
        train_imgs = np.random.rand(10, 32, 32, 3)
        train_labels = np.random.randint(0, 2, 10)
        test_imgs = np.random.rand(5, 32, 32, 3)
        test_labels = np.random.randint(0, 2, 5)

        train_imgs, train_labels, test_imgs, test_labels = self.model.preprocess_data(train_imgs, train_labels, test_imgs, test_labels)

        self.model.initialize_parameters()
        weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = self.model.train(train_imgs, train_labels, test_imgs, test_labels)

        # Check that training has modified weights and bias
        self.assertIsNotNone(weights, "Weights should be updated after training")
        self.assertIsNotNone(bias, "Bias should be updated after training")

        # Check that the loss and accuracy histories are populated
        self.assertGreater(len(loss_history), 0, "Loss history should be populated")
        self.assertGreater(len(test_loss_history), 0, "Test loss history should be populated")
        self.assertGreater(len(accuracy_history), 0, "Accuracy history should be populated")
        self.assertGreater(len(test_accuracy_history), 0, "Test accuracy history should be populated")

if __name__ == '__main__':
    unittest.main()