# 🧠 Multimodal AI-Based Medical Diagnosis System

This project is an AI-powered diagnostic assistant that uses deep learning to analyze medical images of different body parts including Chest X-rays, Brain MRIs, and Knee X-rays. The system predicts potential diseases and provides visual explanations using Grad-CAM heatmaps.

## 🚀 Live Demo
**Gradio App:** [Insert your Gradio link here]  

## 📂 Project Structure
```
├── app.py                 # Main Gradio app script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── report.pdf             # Detailed project report
├── samples/               # Sample X-ray and MRI images
```

## 🧰 Technologies Used
- Python 3.x
- PyTorch
- TorchXRayVision (CheXNet)
- torchvision (ResNet18)
- Gradio (Web Interface)
- OpenCV & Matplotlib (Grad-CAM visualization)

## 🏥 Datasets Used
- **Chest X-rays**: NIH ChestX-ray14 (14 disease classes)
- **Brain MRI**: Kaggle Brain Tumor MRI Dataset (Tumor / No Tumor)
- **Knee X-rays**: Simulated results (future expansion planned)

## 🧠 Models Used
- **CheXNet (DenseNet121)**: For Chest X-ray diagnosis
- **ResNet18**: For Brain Tumor classification
- **Simulated Logic**: For Knee X-rays (demo only)

## 💡 Features
- Multimodal image diagnosis from a single interface
- Grad-CAM overlay for explainability
- Dynamic model switching based on selected body part
- Clean UI using Gradio

## 🛠️ How to Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:7860` in your browser.

## 📈 Future Enhancements
- Add real models for orthopedic diagnosis (e.g., MURA dataset)
- Support DICOM/NIfTI formats for CT and MRI volume scans
- Integrate patient history and clinical notes
- Generate PDF diagnostic reports

## 📚 References
- NIH ChestX-ray14 Dataset
- Rajpurkar et al., CheXNet
- TorchXRayVision GitHub
- Kaggle Brain MRI Dataset
- OpenAI CLIP
- Gradio Documentation

## ✍️ Author
**[DHRUV CHIRANIA]** — Computer Science and Engineering, [MANIPAL UNIVERSITY JAIPUR]  
**Project Guide:** [MS. NEHA SINGH]

Feel free to clone, use, and expand this project!
