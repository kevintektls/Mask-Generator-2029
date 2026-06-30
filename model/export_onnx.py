import torch
import torch.nn as nn
from pathlib import Path

# Importe ton modèle ici (ajuste selon ton projet)
# from train_model import BehavioralCloningCNN 

def export_to_onnx(model_path, output_path, img_height=120, img_width=160):
    # 1. Recréer et charger le modèle
    model = BehavioralCloningCNN() # Remplace par ta classe
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. Créer une fausse entrée (Dummy Input) 
    # Batch=1, Canaux=3 (nos 3 frames stackées), Hauteur, Largeur
    dummy_input = torch.randn(1, 3, img_height, img_width, dtype=torch.float32)

    # 3. Export
    print(f"Exportation du modèle vers {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # Stocke les poids entraînés dans le fichier
        opset_version=17,          # Version stable recommandée
        do_constant_folding=True,  # Optimise les constantes mathématiques
        input_names=['input_frames'],
        output_names=['servo_prediction']
    )
    print("Exportation ONNX réussie ! ✨")

if __name__ == "__main__":
    export_to_onnx("pilot_model.pth", "robot_model.onnx")