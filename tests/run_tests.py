# Import standard libraries
import unittest

# Import third-party libraries
import xmlrunner


if __name__ == '__main__':
    # Only run tests in packages containing an __init__.py file
    unittest.main(module=None,
                  testRunner=xmlrunner.XMLTestRunner(output='test_results'),
                  failfast=False,
                  buffer=False,
                  catchbreak=False,
                  argv=['', 'discover', '-p', '*test.py'])
