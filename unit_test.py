import unittest
from utils import *


# The test case class
class TestFunction(unittest.TestCase):

    # Test method for the add function
    def test_preprocessing(self):
        # Test case: Testing if there only appropriate types
        kidney_df = pd.read_csv(r'datasets\kidney_disease.csv')
        preprocessed_kd_df = preprocessing(kidney_df)
        non_numeric = (preprocessed_kd_df.applymap(type) != float) & (preprocessed_kd_df.applymap(type) != int)
        self.assertEqual(non_numeric.any().any(), 0)  # Asserting if the result is as expected

        # Test case: Testing if there are any Nan Values
        nan = preprocessed_kd_df.isna().any().any()
        self.assertEqual(nan, 0)  # Asserting if the result is as expected

# Running the tests if this script is executed directly
if __name__ == '__main__':
    unittest.main()