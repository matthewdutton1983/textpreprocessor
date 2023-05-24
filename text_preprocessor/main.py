from .pipeline_manager import PreProcessor


def main():
    # Create the instances of PreProcessor and Pipeline
    preprocessor = PreProcessor()
    pipeline = preprocessor.create_pipeline()

    # Add preprocessing methods to the pipeline
    pipeline.add_method([preprocessor.make_lowercase,
                         preprocessor.check_spelling,
                         preprocessor.remove_names,
                         preprocessor.remove_whitespace])

    # Use the pipeline to preprocess the text
    text = ' Helllo, I am John Doe!!! My EMAIL is john.doe@email.com. ViSIT ouR wEbSite www.johndoe.com '
    processed_text = pipeline.execute_pipeline(text)
    print(processed_text)


if __name__ == "__main__":
    main()
