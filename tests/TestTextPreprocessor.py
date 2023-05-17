# Import standard libraries
from unittest import TestCase
from unittest.mock import patch, MagicMock

# Import project code
from text_preprocessor.TextPreprocessor import TextPreprocessor

class TestTextPreprocessor(TestCase):
    def test_replace_words(self):
        input_text = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
        expected_output = 'Helllo, I am John Doe!!! My email is foo.bar@email.com. Visit our website www.johndoe.com'
        replacement_dict = {"john": "foo", "doe": "bar"}
        output_text = TextPreprocessor.replace_words(input_text, replacement_dict=replacement_dict, case_sensitive=True)
        self.assertEqual(output_text, expected_output)


    def test_make_lowercase(self):
        input_text = 'Helllo, I am John Doe!!!'
        expected_output = 'helllo, i am john doe!!!'
        output_text = TextPreprocessor.make_lowercase(input_text)
        self.assertEqual(output_text, expected_output)


    def test_make_uppercase(self):
        input_text = 'Helllo, I am John Doe!!!'
        expected_output = 'HELLLO, I AM JOHN DOE!!!'
        output_text = TextPreprocessor.make_uppercase(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_numbers(self):
        input_text = 'My name is John Doe, I am 30 years old'
        expected_output = 'My name is John Doe, I am  years old'
        output_text = TextPreprocessor.remove_numbers(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_list_markers(self):
        input_text = """
        1. First, we start with initializing our environment.
        2. Next, we import our necessary libraries:
            - pandas for data manipulation
            - matplotlib for data visualization
        3. We then load our data into a pandas DataFrame.
            - This data might come from various sources such as CSV files or databases. 
        4. We clean and preprocess our data.
        5. Finally, we analyze our data and draw conclusions."""
        expected_output = ' First, we start with initializing our environment. Next, we import our necessary libraries:\n    - pandas for data manipulation\n    - matplotlib for data visualization We then load our data into a pandas DataFrame.\n    - This data might come from various sources such as CSV files or databases.  We clean and preprocess our data. Finally, we analyze our data and draw conclusions.\n'
        output_text = TextPreprocessor.remove_list_markers(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_urls(self):
        input_text = 'My name is John Doe, visit my website www.johndoe.com'
        expected_output = 'My name is John Doe, visit my website '
        output_text = TextPreprocessor.remove_urls(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_punctuation(self):
        input_text = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
        expected_output = 'Helllo I am John Doe My email is johndoeemailcom Visit our website wwwjohndoecom'
        output_text = TextPreprocessor.remove_punctuation(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_duplicate_punctuation(self):
        input_text = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
        expected_output = 'Helllo, I am John Doe! My email is john.doe@email.com. Visit our website www.johndoe.com'
        output_text = TextPreprocessor.remove_duplicate_punctuation(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_special_characters(self):
        input_text = """
        Did you know? The @ symbol is called 'ampersat', and the # symbol is often referred to as 'hash' or 'pound'!
        However, in coding, # might denote a 'comment'. Also, don't forget about the ampersand (&), which represents 'and'.
        Here are some other special characters: å¼«¥ª°©ð±§µæ¹¢³¿®ä£"""
        expected_output = '\nDid you know The  symbol is called ampersat and the  symbol is often referred to as hash or pound\nHowever in coding  might denote a comment Also dont forget about the ampersand  which represents and\nHere are some other special characters \n'
        output_text = TextPreprocessor.remove_special_characters(input_text, remove_unicode=True)
        self.assertEqual(output_text, expected_output)


    def test_expand_contractions(self):
        input_text = "I can't wait to deploy this"
        expected_output = 'I cannot wait to deploy this'
        output_text = TextPreprocessor.expand_contractions(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_email_addresses(self):
        input_text = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
        expected_output = 'Helllo, I am John Doe!!! My email is . Visit our website www.johndoe.com'
        output_text = TextPreprocessor.remove_email_addresses(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_phone_numbers(self):
        input_text = 'My phone number is +1-973-932-9426'
        expected_output = 'My phone number is '
        output_text = TextPreprocessor.remove_phone_numbers(input_text)
        self.assertEqual(output_text, expected_output)


    def test_check_spelling(self):
        input_text = 'misisippi'
        expected_output = 'mississippi'
        output_text = TextPreprocessor.check_spelling(input_text)
        self.assertEqual(output_text, expected_output)


    def test_remove_stopwords(self):
        input_text = 'The quick brown fox jumps over the lazy dog sitting in the warm sunlight on a beautiful day'
        expected_output = ['The','quick','brown','fox','jumps','lazy','dog','sitting','warm','sunlight','beautiful','day']
        output_text = TextPreprocessor.remove_stopwords(input_text)
        self.assertEqual(output_text, expected_output)


    def test_tokenize_words(self):
        input_text = 'The quick brown fox jumps over the lazy dog sitting in the warm sunlight on a beautiful day'
        expected_output = ['The','quick','brown','fox','jumps','over','the','lazy','dog','sitting','in','the','warm','sunlight','on','a','beautiful','day']
        output_text = TextPreprocessor.tokenize_words(input_text)
        self.assertEqual(output_text, expected_output)


    def test_tokenize_sentences(self):
        input_text = "Hello there! It's a beautiful day today. Have you seen the movie that just came out? I heard it's fantastic. Let's go see it!"
        expected_output = ['Hello there!',"It's a beautiful day today.",'Have you seen the movie that just came out?',"I heard it's fantastic.","Let's go see it!"]
        output_text = TextPreprocessor.tokenize_sentences(input_text)
        self.assertEqual(output_text, expected_output)

    
    

