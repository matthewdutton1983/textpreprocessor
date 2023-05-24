PreProcessor for NLP
=============

Python package that provides a comprehensive toolkit for pre-processing texual data.

Usage
--------
Install the package using pip:
```bash
pip install text_preprocessor
```

Then, add the package to your python script and call appropriate functions:

```python
from text_preprocessor import PreProcessor

# Create an instance of the PreProcessor class
preprocessor = PreProcessor()

# Preprocess text using individual methods
input_text = 'Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com'

preprocessed_text = preprocessor.remove_email_addresses(input_text)

print(preprocessed_text)

'Hello, my name is Joe Bloggs and my email address is <EMAIL_ADDRESS>.'
```

You can also chain together multiple preprocessing methods and run them all as a pipeline. 
There is a default pipeline that features some of the most common / standard preprocessing tasks.
Or you can create your own custom pipeline.

Note: Pipelines are configured to automatically give preference to string methods, however, this
feature can be overridden to preserve the order in which methods were added to the pipeline.

```python
# Preprocess text using default pipeline
input_text = 'Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com'

pipeline = preprocessor.create_pipeline(load_defaults=True)

result = preprocessor.execute_pipeline(input_text)

print(result)

{'processed_text': ['this is a sample sentence now would you like another one'], 'exceptions_list': []}
```

The output of the pipeline is a dictionary that contains the processed text and an exceptions list. The latter 
contains a list of any methods that failed to run. To access the text simply unmarshall the dictionary as follows:

```python
print(result["processed_text"])

['this is a sample sentence now would you like another one']
```

If the default pipeline does not meet your specific needs, it is easy to create a custom pipeline.

```python
# Preprocess text using custom pipeline 
pipeline = create_pipeline()

pipeline.add_methods([
  preprocessor.make_lowercase,
  preprocessor.remove_whitespace,
  preprocessor.remove_email_addresses
  preprocessor.handle_line_feeds
])


input_text = "  Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com\r\n  "

result = preprocessor.execute_pipeline(input_text)                                        

print(result)

{
  'processed_text': 'hello, my name is joe bloggs and my email address is <EMAIL_ADDRESS>',
  'exceptions_list': []
}
```

Features
--------

| Feature                                                       | Method                                | Default Pipeline
| :------------------------------------------------------------ |:------------------------------------- | :----------------------
| check and correct spellings                                   | check_spelling                        | Yes
| convert to lower case                                         | make_lowercase                        | Yes
| handle line feeds                                             | handle_line_feeds                     | Yes
| lemmatize words                                               | lemmatize_words                       | Yes
| remove punctuations                                           | remove_punctuation                    | Yes
| remove stop words                                             | remove_stopwords                      | Yes
| remove unnecessary whitespace                                 | remove_whitespace                     | Yes
| tokenize sentences                                            | tokenize_sentences                    | Yes
| apply a specific encoding to text                             | encode_text                           | No
| convert to upper case                                         | make_uppercase                        | No
| expand contractions                                           | expand_contractions                   | No
| normalize unicode (e.g., café -> cafe)                        | normalize_unicode                     | No
| remove bullets and numbering                                  | remove_list_markers                   | No
| remove credit card numbers                                    | remove_credit_card_numbers            | No
| remove duplicate punctuations                                 | remove_duplicate_punctuation          | No
| remove email addresses                                        | remove_email_addresses                | No
| remove names                                                  | remove_name                           | No
| remove numbers                                                | remove_numbers                        | No
| remove phone numbers                                          | remove_phone_numbers                  | No
| remove Social Security Numbers                                | remove_ssn                            | No
| remove special characters                                     | remove_special_characters             | No
| remove URLs                                                   | remove_url                            | No
| replace words                                                 | replace_words                         | No
| stem words                                                    | stem_words                            | No
| substitute custom words (e.g., vs -> versus)                  | substitute_token                      | No
| tokenize words                                                | tokenize_words                        | No
