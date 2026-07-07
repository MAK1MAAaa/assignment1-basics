def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
 return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# decode_utf8_bytes_to_str_wrong("原神启动！".encode("utf-8"))

b"\x80\x80".decode("utf-8")
