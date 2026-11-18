import hashlib, json
def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def short_hash(*parts, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:n]
