from line_profiler import LineProfiler
from enum import Enum, unique

EXPORT_PATH = "./data/"

profiler = LineProfiler()

def profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)
    return inner

@unique
class RunType(Enum):
    NORMAL = 1
    PROFILING = 2