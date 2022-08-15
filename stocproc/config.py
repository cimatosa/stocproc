import configparser

config = configparser.ConfigParser()
config.read("stocproc.ini")


USE_NORMALIZED_DIFF = config["FFT"].getboolean("use_normalized_diff", fallback=False)
