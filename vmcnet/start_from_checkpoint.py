"""Example FermiNet VMC script to learn the Boron atom with Adam."""
import logging

import B_adam_ferminet as runner

if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    runner.main(True)
