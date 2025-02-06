import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import joblib
import numpy as np
import cv2
import os

# Funzione per estrarre HOG da un'immagine
def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Usa la stessa dimensione dell'addestramento

    # Configura l'estrattore HOG come in fase di training
    win_size = (128, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(img)

    print(f"Numero di features estratte: {features.shape[0]}")  # Debug per verifica

    return features.flatten().reshape(1, -1)


# Funzione per caricare un modello.pkl
def load_model():
    model_path = filedialog.askopenfilename(filetypes=[("File del modello", "*.pkl")])
    if model_path:
        try:
            global model
            model = joblib.load(model_path)  # Usa joblib invece di pickle
            messagebox.showinfo("Successo", f"Modello caricato con successo: {os.path.basename(model_path)}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare il modello: {e}")

# Caricamento iniziale del modello
model = None
class_names = ['crop', 'weed']  # Aggiungi eventuali classi del tuo modello

def load_and_predict(file_path):
    if not file_path:
        return
    
    if model is None:
        messagebox.showerror("Errore", "Carica prima un modello!")
        return

    try:
        features = extract_hog_features(file_path)
        predicted_class = model.predict(features)[0]
        
        # Update label text
        label_result.config(text=f"Predizione: {class_names[predicted_class]}")
        
        # Load and display original image
        original_image = Image.open(file_path).resize((300, 300))
        original_photo = ImageTk.PhotoImage(original_image)
        label_original.config(image=original_photo)
        label_original.image = original_photo

    except Exception as e:
        messagebox.showerror("Errore", f"Si è verificato un errore: {e}")

# Crea la finestra principale
window = tk.Tk()
window.title("Rilevamento Malattie delle Piante")
window.geometry("850x550")
window.config(bg="#f4f7f6")

# Intestazione
header_frame = tk.Frame(window, bg="#4caf50", pady=10)
header_frame.pack(fill="x")
header_label = tk.Label(header_frame, text="Rilevamento Malattie delle Piante", font=("Helvetica", 20, "bold"), fg="white", bg="#4caf50")
header_label.pack()

# Bottone per caricare il modello
btn_load_model = tk.Button(window, text="Carica Modello", command=load_model,
                           font=("Helvetica", 12), bg="#4caf50", fg="white", relief="flat", width=20)
btn_load_model.pack(pady=10)

# Bottone per caricare l'immagine
btn_load = tk.Button(window, text="Carica Immagine", command=lambda: load_and_predict(filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])),
                     font=("Helvetica", 12), bg="#4caf50", fg="white", relief="flat", width=20)
btn_load.pack(pady=20)

# Label per mostrare la predizione
label_result = tk.Label(window, text="Predizione: ", font=("Helvetica", 14), bg="#f4f7f6")
label_result.pack(pady=10)

# Frame per l'immagine
frame_images = tk.Frame(window, bg="#f4f7f6")
frame_images.pack(pady=20)

# Label per l'immagine originale
label_original = tk.Label(frame_images, text="Immagine Originale", font=("Helvetica", 12), bg="#f4f7f6")
label_original.pack(side=tk.LEFT, padx=20)

# Footer
footer_frame = tk.Frame(window, bg="#4caf50", pady=15)
footer_frame.pack(fill="x", side="bottom")
footer_label = tk.Label(footer_frame, text="© 2025 Rilevamento Malattie delle Piante", font=("Helvetica", 10), fg="white", bg="#4caf50")
footer_label.pack()

# Avvio GUI
window.mainloop()
