# IMPORTS ######################################################################
import os
from pandas import Timestamp
# SCRIPTS ######################################################################
class CustomLogger:
    def __init__(self, foldername : str = None):
        self.name = ""
        self.foldername = foldername

    def initialise_log(self, type : str):
        # Initialise the log file if it doesn't exist
        with open(f"{self.foldername}/{type}.log", "w") as file : 
            file.write(f"### {type} logs ###\n")

    def __call__(self, message, printing : bool = False, type : str = "LOOP_INFO",
        skip_line : str = None) -> None:
        if printing: print(message)

        if f"{type}.log" not in os.listdir(self.foldername):
            self.initialise_log(type)
        
        with open(f"{self.foldername}/{type}.log", "a") as file:
            if skip_line == "before" : file.write("\n")
            file.write(f"[{type}] ({Timestamp.now().strftime('%Y-%m-%d %X')}): "
                      f"{message}\n")
            if skip_line == "after" : file.write("\n")