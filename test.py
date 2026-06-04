# from datetime import datetime,timezone
# now=datetime.now(timezone.utc)
# print(now)
# print(now.isoformat())
# print(type(now))


text=" http://127.0.0.1:8800/ "
text=text.strip()
text=text.rstrip('/')
print(text)

# print(f"原始:'{text}'")
# print(f"使用strip:'{text.strip()}'")
# print(f"使用rstrip:'{text.rstrip()}'")
# print(f"使用rstrip('/'):{text.rstrip('/')}")