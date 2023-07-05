from unittest import TestCase
from test import load_dataset
from test import load_model_test
from test import np
from test import tf
class Test(TestCase):
    def __int__(self):
        self.n_get = 50
        self.n_test = 0


    def get_random_testset(self):
        self.trainX, self.trainY, self.testX, self.testY = load_dataset()
        self.n_test = len(self.testX)
        indices = np.random.choice(self.n_test, self.n_get, replace=False)

        self.rdTestX = self.testX[indices]
        self.rdTestY = self.testY[indices]

    def prep_pixels_convert(self, testX):
        # convert from integers to floats
        test_norm = testX.astype('float32')
        # normalize to range 0-1
        test_norm = test_norm / 255.0
        # return normalized images
        return test_norm

    def test_load_dataset(self):
        self.__int__()
        self.get_random_testset()
        assert len(self.rdTestX) == self.n_get
        assert len(self.rdTestY) == self.n_get

    def test_run_example(self):
        self.__int__()
        self.get_random_testset()

        model = load_model_test()
        self.rdTestX = self.prep_pixels_convert(self.rdTestX)

        _, acc = model.evaluate(self.rdTestX, self.rdTestY, verbose=0)

        assert acc >= 0.98