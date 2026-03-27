"""
SharedMemStore — drop-in replacement for redis.Redis() using /tmp files.
Supports the get / set / ping subset used by run_robot.py and run_motion.py.
No server process required.
"""
import os

_SHM_DIR = "/tmp/g1_shm"
os.makedirs(_SHM_DIR, exist_ok=True)


class SharedMemStore:
    def ping(self):
        return True

    def set(self, key, value):
        if isinstance(value, str):
            value = value.encode()
        path = os.path.join(_SHM_DIR, key)
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(value)
        os.replace(tmp_path, path)  # atomic on Linux — reader sees old or new, never empty

    def get(self, key):
        path = os.path.join(_SHM_DIR, key)
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data if data else None  # guard against empty read
        except FileNotFoundError:
            return None
