import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os

selected_image_path = None 

def on_run(run_type):
    try:
        selected_algorithm = algorithm_var.get()
        selected_k = k_var.get() if k_var.get() != "none" else None
        selected_nr_poze = nr_poze_var.get()  # This is a string, need to convert to int
        selected_norma = norma_var.get()

        # Error handling
        if selected_algorithm == "none":
            messagebox.showerror("Error", "Please select an algorithm.")
            raise ValueError("Please select an algorithm.")
        if selected_nr_poze == "none":
            messagebox.showerror("Error", "Please select a number of training images.")
            raise ValueError("Please select a number of training images.")

        # Convert selected_nr_poze to integer
        selected_nr_poze = int(selected_nr_poze)

        if run_type == "searchImage":
            if not selected_image_path:
                messagebox.showerror("Error", "Please select an image for search.")
                raise ValueError("Please select an image for search.")
            if not selected_k:
                messagebox.showerror("Error", "Please select a k value.")
                raise ValueError("Please select a k value.")
            if selected_norma == "none":
                messagebox.showerror("Error", "Please select a norm.")
                raise ValueError("Please select a norm.")

        print(f"Algorithm: {selected_algorithm}, k: {selected_k}, Training Images: {selected_nr_poze}, Norm: {selected_norma}, Image: {selected_image_path}")

        script_mapping = {
            "kNN": r"C:\Users\win\OneDrive\Desktop\GUI\kNN\kNN.py",
            "Eigenfaces-clase": r"C:\Users\win\OneDrive\Desktop\GUI\Eigenfaces_clase\Eigenfaces-clase.py",
            "Eigenfaces-classic": r"C:\Users\win\OneDrive\Desktop\GUI\Eigenfaces_classic\Eigenfaces-classic.py",
            "Lanczos": r"C:\Users\win\OneDrive\Desktop\GUI\Lanczos\Lanczos.py"
        }

        # Run the selected algorithm
        if selected_algorithm in script_mapping:
            script_name = script_mapping[selected_algorithm]
            if run_type == "searchImage":
                command = f"python {script_name} {selected_nr_poze} searchImage {selected_image_path} {selected_k} {selected_norma}"
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                persoana_gasita = result.stdout.strip().split()[-1]  # Assuming the last item in the output is the person ID
                found_image_path = os.path.join("C:/Users/win/.spyder-py3/att_faces", f"s{persoana_gasita}", "1.pgm")
                display_images(selected_image_path, found_image_path)
            else:
                command = f"python {script_name} {selected_nr_poze}"
                subprocess.run(command, shell=True)

    except ValueError as e:
        print(f"Error: {e}")


def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;.png;.bmp;*.pgm")])
    if selected_image_path:
        print(f"Selected image: {selected_image_path}")

def display_images(searched_image_path, found_image_path):
    # Display the searched image
    searched_image = Image.open(searched_image_path)
    searched_image = searched_image.resize((200, 200))
    searched_image_tk = ImageTk.PhotoImage(searched_image)
    searched_image_label.config(image=searched_image_tk)
    searched_image_label.image = searched_image_tk

    # Display the found image
    found_image = Image.open(found_image_path)
    found_image = found_image.resize((200, 200))
    found_image_tk = ImageTk.PhotoImage(found_image)
    found_image_label.config(image=found_image_tk)
    found_image_label.image = found_image_tk

root = tk.Tk()
root.title("Algorithm Selector")
root.geometry("1200x700")
root.resizable(False, False)
root.configure(bg='#f0f8ff')  # Light cyan background color
center_window(root)

label_font = ("Helvetica", 12, "bold")
button_font = ("Helvetica", 12, "bold")
frame_bg = "#8fa7c4"  # Light blue-gray background for frames

# Training images section
nr_poze_label_frame = tk.Frame(root, bg=frame_bg, bd=2, relief="groove")
nr_poze_label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")
nr_poze_label = tk.Label(nr_poze_label_frame, text="Training Images:", font=label_font, bg=frame_bg, fg="white")
nr_poze_label.pack(padx=5, pady=5)

nr_poze_var = tk.StringVar(value="none")
nr_poze_frame = tk.Frame(root, bg='#f0f8ff')
nr_poze_frame.grid(row=0, column=1, padx=10, pady=10, sticky="w")

nr_poze_values = ["6", "8", "9"]
nr_poze_percentages = ["60% training - 40% testing", "80% training - 20% testing", "90% training - 10% testing"]

for i, nr_poze in enumerate(nr_poze_values):
    frame = tk.Frame(nr_poze_frame, bg='#f0f8ff')
    frame.pack(anchor="w")
    rb = tk.Radiobutton(frame, text=nr_poze, variable=nr_poze_var, value=nr_poze, bg='#f0f8ff', font=label_font)
    rb.pack(side="left")
    lbl = tk.Label(frame, text=nr_poze_percentages[i], bg='#f0f8ff', font=label_font)
    lbl.pack(side="left", padx=5)

# Algorithm selection section
algorithm_label_frame = tk.Frame(root, bg=frame_bg, bd=2, relief="groove")
algorithm_label_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")
algorithm_label = tk.Label(algorithm_label_frame, text="Algorithm:", font=label_font, bg=frame_bg, fg="white")
algorithm_label.pack(padx=5, pady=5)

algorithm_var = tk.StringVar(value="none")
algorithm_frame = tk.Frame(root, bg='#f0f8ff')
algorithm_frame.grid(row=1, column=1, padx=10, pady=10, sticky="w")

algorithms = ["kNN", "Eigenfaces-classic", "Eigenfaces-clase", "Lanczos"]
k_values = ["1","3","5","7","9","20", "40", "60", "80", "100"]

# Functia pentru a actualiza valorile din combobox in functie de algoritm
def update_k_values(*args):
    selected_algorithm = algorithm_var.get()

    # Determinarea valorilor pentru k in functie de algoritm
    if selected_algorithm == "kNN":
        k_values = ["1", "3", "5", "7", "9"]
    else:
        k_values = ["20", "40", "60", "80", "100"]

    # Actualizează valorile din combobox
    k_dropdown['values'] = k_values
    k_dropdown.current(0)  # Setează valoarea implicită pe primul element

# Leagă funcția de variabila algorithm_var
algorithm_var.trace("w", update_k_values)

for alg in algorithms:
    rb = tk.Radiobutton(algorithm_frame, text=alg, variable=algorithm_var, value=alg, bg='#f0f8ff', font=label_font)
    rb.pack(anchor="w")

# k value selection section
k_label_frame = tk.Frame(root, bg=frame_bg, bd=2, relief="groove")
k_label_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")
k_label = tk.Label(k_label_frame, text="k:", font=label_font, bg=frame_bg, fg="white")
k_label.pack(padx=5, pady=5)

k_var = tk.StringVar()
k_dropdown_frame = tk.Frame(root, bg='#f0f8ff')
k_dropdown_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")
k_dropdown = ttk.Combobox(k_dropdown_frame, textvariable=k_var)
k_dropdown['values'] = k_values
k_dropdown.current(0)
k_dropdown.pack()

# Norm selection section
norma_label_frame = tk.Frame(root, bg=frame_bg, bd=2, relief="groove")
norma_label_frame.grid(row=3, column=0, padx=10, pady=10, sticky="w")
norma_label = tk.Label(norma_label_frame, text="Norm:", font=label_font, bg=frame_bg, fg="white")
norma_label.pack(padx=5, pady=5)

norma_var = tk.StringVar(value="none")
norma_frame = tk.Frame(root, bg='#f0f8ff')
norma_frame.grid(row=3, column=1, padx=10, pady=10, sticky="w")

norma_values = ["1 (Manhattan)", "2 (Euclidean)", "inf (Infinity)", "cos (Cosinus)"]

for norma in norma_values:
    rb = tk.Radiobutton(norma_frame, text=norma, variable=norma_var, value=norma, bg='#f0f8ff', font=label_font)
    rb.pack(anchor="w")

# Buttons for selecting image and running the search or showing stats
run_button = tk.Button(root, text="Select a search image", command=select_image, font=button_font, bg="#4CAF50", fg="white", activebackground="#45a049")
run_button.grid(row=4, column=0, columnspan=1, padx=10, pady=20)

run_button = tk.Button(root, text="Search image for this config", command=lambda: on_run("searchImage"), font=button_font, bg="#009688", fg="white", activebackground="#00796b")
run_button.grid(row=5, column=0, columnspan=1, padx=10, pady=20)

run_button = tk.Button(root, text="Show statistics for this config (training + alg)", command=lambda: on_run("showStats"), font=button_font, bg="#00796b", fg="white", activebackground="#004d40")
run_button.grid(row=6, column=0, columnspan=1, padx=10, pady=10)

# Create frames for displaying images
image_frame = tk.Frame(root, bg='#f0f8ff')
image_frame.grid(row=0, column=2, rowspan=7, padx=10, pady=10, sticky="n")

# Frame for searched image
searched_image_frame = tk.Frame(image_frame, bg='#f0f8ff')
searched_image_frame.pack(pady=10)
searched_image_label_text = tk.Label(searched_image_frame, text="Searched Image", font=label_font, bg='#f0f8ff')
searched_image_label_text.pack()
searched_image_label = tk.Label(searched_image_frame, bg='#f0f8ff')
searched_image_label.pack()

# Frame for found image
found_image_frame = tk.Frame(image_frame, bg='#f0f8ff')
found_image_frame.pack(pady=10)
found_image_label_text = tk.Label(found_image_frame, text="Found Image", font=label_font, bg='#f0f8ff')
found_image_label_text.pack()
found_image_label = tk.Label(found_image_frame, bg='#f0f8ff')
found_image_label.pack()

root.mainloop()
