# Q-Vision/qvision/photon_detector.py

import numpy as np


class PhotonDetector:
    def __init__(self, efficiency=1.0, dead_time=0.0, resolving_power=1):
        """
        Initialize a photon detector with given properties.

        efficiency: Probability of detecting a photon (between 0 and 1).
        dead_time: Time during which the detector cannot detect another photon (in arbitrary time units).
        resolving_power: Maximum number of photons the detector can resolve at once.
        """
        self.efficiency = efficiency
        self.dead_time = dead_time
        self.resolving_power = resolving_power
        self.last_detection_time = -np.inf

    def detect(self, num_photons, current_time):
        """
        Simulate the detection of photons.

        num_photons: Number of incoming photons.
        current_time: The current time in the simulation.

        Returns the number of detected photons.
        """
        if current_time - self.last_detection_time < self.dead_time:
            return 0  # Detector is in dead time

        detected_photons = np.random.binomial(n=num_photons, p=self.efficiency)
        self.last_detection_time = current_time

        if detected_photons > self.resolving_power:
            detected_photons = self.resolving_power

        return detected_photons
