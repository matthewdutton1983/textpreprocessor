# Import standard libraries
from unittest import TestCase
from unittest.mock import patch, MagicMock

# Import project code
from text_preprocessor.TextPreprocessor import TextPreprocessor

class TestTextPreprocessor(TestCase):
    def test_make_lowercase(self):
        # Setup
        input_text = 'HellO'
        expected_output = 'hello'
        # Actual call
        output_text = TextPreprocessor.make_lowercase(input_text)
        # Asserts
        self.assertEqual(output_text, expected_output)
