import regex

CONTROL_CHARS_PATTERN = regex.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u00a0]+")
DEFAULT_END_TOKEN = "</s>"
PREDICT_STAGE_NAME = "entire_predict"

# Key should be hidden from other services on the same instance, so it can't be an env var. Should be used in obfuscated code only
ENCRYPTION_KEY = bytes.fromhex('fd4f1f92d78dce3f7e35d8f8ca169db2ee23eb00ffcc696a631e570c27b52982')
