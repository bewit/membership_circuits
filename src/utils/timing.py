import time                                                

def measure_time(f):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()

        print(f"{f.__name__}: {te-ts:.4f}s")
        return result

    return timed