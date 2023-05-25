# Text PreProcessor for NLP

This Python package provides a comprehensive toolkit for pre-processing texual data for use in Natural Language Processing (NLP) projects.

## Usage

Install the package using pip:

```bash
pip install text_preprocessor
```

Once the package is installed, add it to your script and call appropriate functions:

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

preprocessor = PreProcessor(default_pipeline=True)

result = preprocessor.execute_pipeline(input_text)

print(result)

'this is a sample sentence now would you like another one'
```

If the default pipeline does not meet your specific needs, it is easy to create a custom pipeline.

```python
# Preprocess text using custom pipeline
pipeline.add_methods([
  preprocessor.make_lowercase,
  preprocessor.remove_whitespace,
  preprocessor.remove_email_addresses
  preprocessor.handle_line_feeds
])

input_text = "  Hello, my name is Joe Bloggs and my email address is joe.bloggs@email.com\r\n  "

result = preprocessor.execute_pipeline(input_text)

print(result)

'hello, my name is joe bloggs and my email address is <EMAIL_ADDRESS>'
```

## Features

| Feature                                      | Method                       | Default Pipeline |
| :------------------------------------------- | :--------------------------- | :--------------- |
| check and correct spellings                  | check_spelling               | Yes              |
| convert to lower case                        | make_lowercase               | Yes              |
| handle line feeds                            | handle_line_feeds            | Yes              |
| lemmatize words                              | lemmatize_words              | Yes              |
| remove punctuations                          | remove_punctuation           | Yes              |
| remove stop words                            | remove_stopwords             | Yes              |
| remove unnecessary whitespace                | remove_whitespace            | Yes              |
| tokenize sentences                           | tokenize_sentences           | Yes              |
| apply a specific encoding to text            | encode_text                  | No               |
| convert numbers to words                     | numbers_to_words             | No               |
| convert to upper case                        | make_uppercase               | No               |
| expand contractions                          | expand_contractions          | No               |
| normalize unicode (e.g., café -> cafe)       | normalize_unicode            | No               |
| remove bullets and numbering                 | remove_list_markers          | No               |
| remove credit card numbers                   | remove_credit_card_numbers   | No               |
| remove duplicate punctuations                | remove_duplicate_punctuation | No               |
| remove email addresses                       | remove_email_addresses       | No               |
| remove names                                 | remove_name                  | No               |
| remove numbers                               | remove_numbers               | No               |
| remove phone numbers                         | remove_phone_numbers         | No               |
| remove Social Security Numbers               | remove_ssn                   | No               |
| remove special characters                    | remove_special_characters    | No               |
| remove URLs                                  | remove_url                   | No               |
| replace words                                | replace_words                | No               |
| stem words                                   | stem_words                   | No               |
| substitute custom words (e.g., vs -> versus) | substitute_token             | No               |
| tokenize words                               | tokenize_words               | No               |
