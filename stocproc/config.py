import configparser

config = configparser.ConfigParser()
config.read("stocproc.ini")


USE_NORMALIZED_DIFF = config.getboolean("FFT", "use_normalized_diff", fallback=False)
