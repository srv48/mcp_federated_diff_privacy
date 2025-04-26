import psutil

def get_system_context():
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    return {"cpu": cpu_usage, "memory": mem_usage}

def select_model_based_on_context(context):
    if context["cpu"] > 70 or context["memory"] > 80:
        return "small"
    else:
        return "large"