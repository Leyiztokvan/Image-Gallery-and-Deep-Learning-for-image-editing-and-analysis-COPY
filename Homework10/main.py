"""
This is the main entry point for the Python project. It provides a simple command-line
interface that allows users to choose which part of the project to run: either the CNN module
or the Image Gallery Examples. The script facilitates easy navigation between different functionalities
of the project without needing to directly interact with the individual scripts.
"""

import sys


def main_menu():
    """
    Displays the main menu options to the user and reads their input.

    Returns:
        choice (str): The user's menu choice as a string.
    """
    print("\nWelcome to our Python project!")
    print("Please choose which module to run. If you run our app for the first time, please start with 1. Image Gallery Examples!:")
    print("1. Image Gallery Examples")
    print("2. CNN Module")
    print("3. Exit")

    choice = input("Enter your choice (1-3): ")
    return choice


def main():
    """
    Main function that loops continuously to provide a menu-driven interface.
    Depending on the user's choice, it executes the corresponding Python script.
    """
    while True:
        choice = main_menu()
        if choice == '2':
            print("\nRunning CNN Module...\n")
            try:
                # Executes the CNN.py script using exec() function, which allows running a Python script dynamically
                exec(open("CNN.py").read(), globals())
            except Exception as e:
                print(f"An error occurred: {e}")
        elif choice == '1':
            print("\nRunning Image Gallery Examples...\n")
            try:
                # Executes the Img_Gallery_Examples.py script using exec() function
                exec(open("Img_Gallery_Examples.py").read(), globals())
            except Exception as e:
                print(f"An error occurred: {e}")
        elif choice == '3':
            print("Exiting the program...")
            sys.exit()
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()
