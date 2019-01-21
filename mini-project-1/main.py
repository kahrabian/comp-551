import logging
from datetime import datetime


def setup_logger():
    logging.basicConfig(filename=datetime.now().strftime('./logs/%Y-%m-%d-%S.log'), level=logging.INFO)


def main():
    pass


if __name__ == '__main__':
    main()
