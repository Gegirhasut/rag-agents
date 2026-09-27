import os

# Тесты не должны зависеть от локального .env и реальных ключей
os.environ.setdefault("APP_ENV", "test")
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
