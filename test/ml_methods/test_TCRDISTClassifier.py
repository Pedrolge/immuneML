import os
import pickle
import shutil
from unittest import TestCase

import dill
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from source.caching.CacheType import CacheType
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.TCRDISTClassifier import TCRDISTClassifier
from source.util.PathBuilder import PathBuilder


class TestTCRDISTClassifier(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_data(self):
        x = np.array([[0., 1., 2., 3.],
                      [1., 0., 1., 2.],
                      [2., 1., 0., 1.],
                      [3., 2., 1., 0.]])
        y = {"test": np.array([0, 0, 1, 1])}

        return x, y, EncodedData(examples=x, labels=y)

    def test_fit(self):
        x, y, encoded_data = self._prepare_data()
        knn = TCRDISTClassifier(percentage=0.75)
        knn.fit(encoded_data, y, ["test"])
        predictions = knn.predict(encoded_data)
        self.assertTrue(np.array_equal(y["test"], predictions["test"]))

        encoded_data.examples = np.array([[1.1, 0.1, 0.9, 1.9]])
        predictions = knn.predict(encoded_data)
        self.assertTrue(np.array_equal([0], predictions["test"]))

    def test_store(self):
        x, y, encoded_data = self._prepare_data()

        cls = TCRDISTClassifier(0.75)
        cls.fit(encoded_data, y, label_names=["test"])

        path = EnvironmentSettings.root_path + "test/tmp/tcrdist_classifier/"

        cls.store(path)
        self.assertTrue(os.path.isfile(path + "tcrdist_classifier.pickle"))

        with open(path + "tcrdist_classifier.pickle", "rb") as file:
            cls2 = pickle.load(file)

        self.assertTrue(isinstance(cls2["test"], KNeighborsClassifier))

        shutil.rmtree(path)

    def test_load(self):
        x, y, encoded_data = self._prepare_data()

        cls = TCRDISTClassifier(0.75)
        cls.fit(encoded_data, y, label_names=["test"])

        path = PathBuilder.build(EnvironmentSettings.root_path + "test/tmp/tcrdist_classifier_load/")

        with open(path + "tcrdist_classifier.pickle", "wb") as file:
            dill.dump(cls.get_model(), file)

        cls2 = TCRDISTClassifier(percentage=1.)
        cls2.load(path)

        self.assertTrue(isinstance(cls2.get_model()["test"], KNeighborsClassifier))
        self.assertTrue(isinstance(cls2, TCRDISTClassifier))
        self.assertEqual(3, cls2.models['test'].n_neighbors)

        shutil.rmtree(path)