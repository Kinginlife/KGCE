 
import importlib
import inspect
import pkgutil


def get_instances(package, class_type):
    instance_dict = {}
    # Iterate through all modules in the specified package
    for _, name, ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if ispkg:
            continue  # Skip subpackages
        module = importlib.import_module(name)
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, class_type):
                instance_dict[name] = obj
    return instance_dict


def get_benchmarks_environments():
    from kgce import BenchmarkConfig, EnvironmentConfig, benchmarks, environments

    benchmark_configs = get_instances(benchmarks, BenchmarkConfig)
    environment_configs = get_instances(environments, EnvironmentConfig)

    return benchmark_configs, environment_configs
