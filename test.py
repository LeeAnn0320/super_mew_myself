from datetime import datetime,timezone
now=datetime.now(timezone.utc)
print(now)
print(now.isoformat())
print(type(now))