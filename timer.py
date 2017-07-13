import time


class Timer(object):
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print(self._message + ": {:.3f} sec".format(time.time() - self._startTime))
