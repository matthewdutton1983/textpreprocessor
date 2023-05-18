TextPreprocessor for NLP
=============

Python package that provides a comprehensive toolkit for pre-processing texual data.

Usage
--------
Install the package using pip:
```bash
pip install <TBD>
```

Then, add the package to your python script and call appropriate functions:

```python
from text_preprocessing.TextPreprocessor import TextPreprocessor

# Create an instance of the TextPreprocessor class
preprocessor = TextPreprocessor()

# Preprocess text using individual methods
input_text = 'Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com'
preprocessed_text = preprocessor.remove_email_addresses(input_text)
print(preprocessed_text)

'hello, my name is Joe Bloggs and my email address is <EMAIL_ADDRESS>.'
```

You can also run text through a pipeline of different preprocessing methods. 
There is a default pipeline that features some of the most common / standard preprocessing tasks
Or you can create your own custom pipeline.

Note: Pipelines are configured to automatically give preference to string methods, however, this
feature can be overridden to preserve the order in which methods were added to the pipeline.

```python
# Preprocess text using default pipeline
input_text = 'Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com'
pipeline = preprocessor.load_default_pipeline()
preprocessed_text = preprocessor.execute_pipeline(input_text, pipeline)
print(preprocessed_text)

'output to be added'

# Preprocess text using custom preprocess functions in the pipeline 
input_text = 'Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com'
pipeline = [] # You can also run preprocessor.clear_pipeline(pipeline)
pipeline = preprocessor.add_to_pipeline([preprocessor.make_lowercase, 
                                        preprocessor.remove_whitespace, 
                                        preprocessor.remove_stopwords, 
                                        preprocessor.stem_words, 
                                        preprocessor.tokenize_sentences])
preprocessed_text = preprocessor.execute_pipeline(input_text, pipeline)                                        
print(preprocessed_text)

'output to be added'
```

Features
--------

| Feature                                                       | Method                                | Default Pipeline
| :------------------------------------------------------------ |:------------------------------------- | :----------------------
| replace words                                                 | replace_words                         | 
| convert to lower case                                         | make_lowercase                        | Yes
| convert to upper case                                         | make_uppercase                        |
| remove numbers                                                | remove_numbers                        |
| remove bullets and numbering                                  | remove_list_markers                   |
| remove URLs                                                   | remove_url                            |
| remove punctuations                                           | remove_punctuation                    | Yes
| remove just duplicate punctuations                            | remove_duplicate_punctuation          |
| remove special characters                                     | remove_special_characters             |
| remove email addresses                                        | remove_email_addresses                |
| remove stop words                                             | remove_stopwords                      | Yes
| remove names                                                  | remove_name                           |
| remove unnecessary whitespace                                 | remove_whitespace                     | Yes
| remove phone numbers                                          | remove_phone_numbers                  |
| remove Social Security Numbers                                | remove_ssn                            |
| remove credit card numbers                                    | remove_credit_card_numbers            |
| expand contractions                                           | expand_contractions                   |
| check and correct spellings                                   | check_spelling                        | Yes
| tokenize words                                                | tokenize_words                        |
| tokenize sentences                                            | tokenize_sentences                    | Yes
| normalize unicode (e.g., café -> cafe)                        | normalize_unicode                     |
| apply a specific encoding to text                             | encode_text                           |
| stem words                                                    | stem_words                            |
| lemmatize words                                               | lemmatize_words                       | Yes
| substitute custom words (e.g., vs -> versus)                  | substitute_token                      |
| handle line feeds                                             | handle_line_feeds                     | Yes
