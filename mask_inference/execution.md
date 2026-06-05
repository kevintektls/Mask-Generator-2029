cd /home/nolhan/Delivery-2026/Mask-Generator-2029/mask_inference

# Avec images par défaut (./test_images)
python inference_realtime.py

# Ou spécifier un dossier
python inference_realtime.py --input /chemin/vers/images --model ../model/unet_scripted.pt

Charge images depuis un dossier (./test_images par défaut)

Pas de dépendance DepthAI (juste OpenCV, NumPy, PyTorch)

Affiche image | masque côte à côte

Navigation : n = image suivante, p = image précédente ✅ s = sauvegarder snapshot ✅ q = quitter