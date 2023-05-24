from .pipeline_manager import PreProcessor


def main():
    preprocessor = PreProcessor()
    pipeline = preprocessor.create_pipeline(load_defaults=True)
    text = ' Helllo, I am John Doe!!! My EMAIL is john.doe@email.com. ViSIT ouR wEbSite www.johndoe.com '
    result = pipeline.execute_pipeline(text)
    print(result)

if __name__ == "__main__":
    main()
