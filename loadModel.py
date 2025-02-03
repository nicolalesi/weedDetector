import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os

# Funzione per caricare un modello
def load_model():
    model_path = filedialog.askopenfilename(filetypes=[("File del modello", "*.h5;*.pkl")])
    if model_path:
        try:
            global model
            model = tf.keras.models.load_model(model_path)
            messagebox.showinfo("Successo", f"Modello caricato con successo: {os.path.basename(model_path)}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare il modello: {e}")

# Caricamento iniziale del modello
model = None
class_names = ['crop', 'weed']  # Aggiungi eventuali classi del tuo modello

def process_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(139, 139))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (139, 139))
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def load_and_predict(file_path):
    if not file_path:
        return

    if model is None:
        messagebox.showerror("Errore", "Carica prima un modello!")
        return

    try:
        img_array = process_image(file_path) 
        predicted_class = np.argmax(model.predict(img_array)[0])

        # Update label text
        label_result.config(text=f"Predizione: {class_names[predicted_class]}")

        # Load and display original image
        original_image = Image.open(file_path).resize((300, 300))
        original_photo = ImageTk.PhotoImage(original_image)
        label_original.config(image=original_photo)
        label_original.image = original_photo

    except Exception as e:
        messagebox.showerror("Errore", f"Si è verificato un errore: {e}")

# Create the main window
window = tk.Tk()
window.title("Rilevamento Malattie delle Piante")
window.geometry("850x550")
window.config(bg="#f4f7f6")

# Add a header frame
header_frame = tk.Frame(window, bg="#4caf50", pady=10)
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="Rilevamento Malattie delle Piante", font=("Helvetica", 20, "bold"), fg="white", bg="#4caf50")
header_label.pack()

# Button to load model
btn_load_model = tk.Button(window, text="Carica Modello", command=load_model,
                           font=("Helvetica", 12), bg="#4caf50", fg="white", relief="flat", width=20)
btn_load_model.pack(pady=10)

# Button to load image
btn_load = tk.Button(window, text="Carica Immagine", command=lambda: load_and_predict(filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])),
                     font=("Helvetica", 12), bg="#4caf50", fg="white", relief="flat", width=20)
btn_load.pack(pady=20)

# Label to display prediction result
label_result = tk.Label(window, text="Predizione: ", font=("Helvetica", 14), bg="#f4f7f6")
label_result.pack(pady=10)

# Frame for images
frame_images = tk.Frame(window, bg="#f4f7f6")
frame_images.pack(pady=20)

# Labels to display original image
label_original = tk.Label(frame_images, text="Immagine Originale", font=("Helvetica", 12), bg="#f4f7f6")
label_original.pack(side=tk.LEFT, padx=20)

# Add footer frame with a little more space
footer_frame = tk.Frame(window, bg="#4caf50", pady=15)
footer_frame.pack(fill="x", side="bottom")

footer_label = tk.Label(footer_frame, text="© 2025 Rilevamento Malattie delle Piante", font=("Helvetica", 10), fg="white", bg="#4caf50")
footer_label.pack()

# Start the GUI event loop
window.mainloop()
