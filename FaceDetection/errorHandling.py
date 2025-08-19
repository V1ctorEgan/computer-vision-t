def lookup(my_list, index):
    try:
        value = my_list[index]
    except IndexError:
        return "Error: Index out of range."
    else:
        return f"The value at index {index} is {value}."


print(lookup(my_list=["a", "b", "c"], index=2))  # Index in range
print(lookup(my_list=["a", "b", "c"], index=5))  # Index out of range

# task 1
def divide_numbers(numerator, denominator):
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        return "Error: Cannot divide by zero."
    else:
        return f"The result of {numerator} divided by {denominator} is {result}."

print(divide_numbers(numerator=10, denominator=2))  # Valid division
print(divide_numbers(numerator=10, denominator=0))  # Division by zero

# task 2
users = {
    342: "Kwame Nkrumah",
    102: "Nguyen Thi Linh",
    423: "Muhammad bin Abdullah",
    654: "Fatou Diop",
    976: "Diana Martinez",
}


def find_user_name(user_id, users):
    try:
        user_name = users[user_id]
    except KeyError:
        return f"Error: User ID {user_id} not found."
    else:
        return f"User ID {user_id} corresponds to {user_name}."

print(find_user_name(user_id=654, users=users))
print(find_user_name(user_id=999, users=users))

        # using tuple unpacking 
for n in range(1, 11):
    print(f"Calling the function for the {n}th time")
    # Python tuple unpacking with * has been added on the left-hand side
    first, second, *rest = unpredictable_function()

    # Handle the common case (always present)
    print(f"    First item: {first}")
    print(f"    Second item: {second}")

    # Check if there's a third item
    if rest:
        print(f"    Third item: {rest[0]}")
    else:
        print("    No third item present")

# task 3
def print_student_info(student_name, students):
    age, *grade = get_student_info(student_name, students)
    
    print(f"{student_name} is {age} years old.")
    if grade:
        print(f"{student_name} earned a {grade}.")
    

print_student_info(student_name="Aisha", students=students)
print_student_info(student_name="Carlos", students=students)
