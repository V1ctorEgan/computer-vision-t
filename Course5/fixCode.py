import os
result = os.system("pwd")
# task 1
result = os.system("ls") 
print(f"Exit status: {result}")

result = os.system("echo 'Hello, World! 🌍' > hello_world.txt")
print(f"Exit status: {result}")

# Let's change the file permissions to make the file read-only. 
os.system("chmod 444 hello_world.txt")

# We can change the file permissions to grant the necessary rights to perform the action, allowing us to run the commands successfully.
os.system("chmod 666 hello_world.txt")
os.system("echo 'Good night, moon! 🌑' >> hello_world.txt")

# Remove file
os.system("rm -f 'application.log'")
# Create new file
os.system("touch 'application.log'")
# Append entry to log file
os.system("echo 'Log entry 1' >> application.log")
# Append another entry to log file
os.system("echo 'Log entry 2' >> application.log")
# Display file contents
os.system("cat application.log")
# Change permissions to remove read access
os.system("chmod 000 application.log")
# Try to display file contents
os.system("cat application.log")

# task 2
os.system("chmod 666 application.log ")

# Display the contents of the file
os.system("cat application.log")



import subprocess

result = subprocess.run(["ls"], capture_output=True, text=True)

print("Output of 'ls':")
print(result.stdout)
# task 3
filename = "worldquant.txt"
result = subprocess.run(["cat", filename], capture_output=True, text=True)


print(f"Contents of '{filename}':")
print(result.stdout)