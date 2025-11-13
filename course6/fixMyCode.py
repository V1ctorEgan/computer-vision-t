import os

from dotenv import load_dotenv, set_key, unset_key
for key, value in os.environ.items():
    print(f"{key}: {value}")

print(os.environ["DATASET_NAME"])
set_key(
    dotenv_path=".env", key_to_set="API_KEY", value_to_set="your_sample_api_key_here"
)
# The new key-value pair must be reload from .env
load_dotenv()

# Use the key to find the value
print(os.environ["API_KEY"])
unset_key(dotenv_path=".env", key_to_unset="API_KEY")

# task 1
set_key(dotenv_path=".env", key_to_set="PASSWORD", value_to_set="qwerty")

load_dotenv()

print(os.environ["PASSWORD"])

# Remove example key
unset_key(dotenv_path=".env", key_to_unset="PASSWORD")
# task 2
set_key(
    dotenv_path=".env",
    key_to_set="DATABASE_URL",
    value_to_set="your_sample_database_url",
)

load_dotenv()

print(os.environ["DATABASE_URL"])


# Open the .env file
with open(".env", "r") as file:
    lines = file.readlines()

# Filter out the line with 'DATABASE URL'
with open(".env", "w") as file:
    for line in lines:
        if not line.startswith("DATABASE URL"):
            file.write(line)


# task 3
