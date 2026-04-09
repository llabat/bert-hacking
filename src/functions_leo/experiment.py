import hashlib
import json


def create_hash(params):
    """
    Create a stable hash from a dictionary of parameters.
    """
    params_string = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(params_string.encode("utf-8")).hexdigest()