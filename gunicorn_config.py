import multiprocessing

# Worker settings optimized for memory
workers = 1  # Single worker to prevent memory duplication
threads = 2
worker_class = 'gthread'
timeout = 30
max_requests = 50
max_requests_jitter = 5
preload_app = True

# Memory optimizations
worker_tmp_dir = '/dev/shm'
worker_max_requests = 50
worker_max_requests_jitter = 10

def on_starting(server):
    import gc
    gc.collect()
