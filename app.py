import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

# Path to trained YOLO model
MODEL_PATH = "runs/detect/train7/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at `{MODEL_PATH}`. Please ensure the model path is correct.")

# Load the YOLO model
model = YOLO(MODEL_PATH)

def annotate_image(image, detections, min_conf):
    count = 0
    for det in detections:
        if det.boxes is None:
            continue
        for box in det.boxes:
            confidence = box.conf[0].cpu().numpy()
            if confidence < min_conf:
                continue
            count += 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"Elephant {confidence:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image, count

def detect_elephants(image, min_conf):
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    start_time = time.time()
    results = model(image_cv, conf=min_conf)
    inference_time = time.time() - start_time
    
    annotated_image, total_count = annotate_image(image_cv.copy(), results, min_conf)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_image)
    
    return annotated_pil, total_count, round(inference_time, 2)
# Load poached elephant images
poached_images = ["D:\elephant-guard\img\er2.jpg", "D:\elephant-guard\img\er3.jpg", "D:\elephant-guard\img\er4.jpg", "D:\elephant-guard\img\er5.jpg"]
loaded_poached_images = [Image.open(img) for img in poached_images if os.path.exists(img)]

# Custom CSS styling
custom_css = """
body {
    background-color: white !important;
    color: black !important;
}
.gallery {
    background: white !important;
}
"""

# Project documentation
project_docs = """


### **📌 Project Description**
ElephantGuard is an **AI-powered conservation system** designed to **detect, track, and prevent poaching** in wildlife reserves using computer vision and reinforcement learning. The system leverages **drone-based aerial imagery** to **identify elephants and potential poaching threats in real-time**, significantly reducing the cost and inefficiencies of traditional **manual surveys** conducted via helicopters or planes. 

By integrating advanced deep learning models with **multi-agent reinforcement learning (MARL)**, we aim to **autonomously coordinate drones** for large-scale monitoring, helping park rangers take proactive measures against illegal poaching.

### **💡 Motivation**
- **Poaching Crisis:** Elephant poaching remains a **critical threat** due to high demand for ivory.
- **Manual Tracking Limitations:** Current elephant tracking methods rely on **expensive and slow** aerial surveys using planes and helicopters.
- **Delayed Detection:** Most poaching activities are detected **too late**, making intervention difficult.
- **Cost-Efficient Alternative:** Our AI-powered **drone-based surveillance system** aims to provide a **faster, cheaper, and more effective** solution.

---

### **🚀 Project Status**
#### ✅ **Current Implementation (Completed)**
- **Computer Vision System**
  - **YOLOv11 model** trained on 9,600+ aerial images
  - Specialized in **drone-captured imagery detection**
  - Achieved **93.2% mAP@0.5** on the validation set
  - Capable of **real-time elephant detection**

#### 🚧 **In Development**
- **Reinforcement Learning Framework**
  - Researching **Multi-Agent Reinforcement Learning (MARL)**
  - Designing **drone coordination strategies**
  - Developing a **simulation environment** using OpenAI Gym

#### 🔮 **Future Roadmap**
1. Integrate **CV system with RL framework** for drone coordination
2. Develop a **real-time alert system** for park rangers
3. Implement **heatmap analysis** for optimizing patrol routes
4. Expand detection capabilities to include **vehicles & poaching activities**


### **🌍 How You Can Help**
We’re actively looking for:
- **MARL experts** for reinforcement learning guidance
- **Drone footage contributors** for real-world model testing
- **Conservation partners** to help us deploy ElephantGuard in the field
- **GPU compute resources** to enhance model training efficiency

📂 [GitHub Repository](https://github.com/sudhersankv/elephant-guard)
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# <center>🐘 ElephantGuard</center>")
    gr.Markdown("### <center>AI Conservation System for Elephant Protection</center>")
    
    with gr.Tabs():
        with gr.TabItem("About"):
            
            
            # Poached elephants gallery
            gr.Markdown("## 💔 The Tragic Reality of Elephant Poaching")
            gr.Markdown("### *Silent giants, stolen too soon...*")
            gr.Gallery(
                value=loaded_poached_images,
                label="Poached Elephants",
                height=400,
                preview=True,
                allow_preview=True
            )
            gr.Markdown(project_docs)

        
        with gr.TabItem("Live Detection"):
            gr.Markdown("### **🔍 Upload an Aerial Image for Elephant Detection**")
            with gr.Row():
                with gr.Column(scale=1):
                    upload_box = gr.Image(label="Upload Aerial Image", type="pil", height=400)
                    with gr.Accordion("Advanced Settings", open=False):
                        min_conf_slider = gr.Slider(0.0, 1.0, value=0.25, label="Detection Confidence Threshold")
                    detect_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Detection Results", height=400)
                    with gr.Row():
                        count_box = gr.Textbox(label="Elephants Detected")
                        time_box = gr.Textbox(label="Processing Time")
            
            detect_btn.click(
                fn=detect_elephants,
                inputs=[upload_box, min_conf_slider],
                outputs=[output_image, count_box, time_box]
            )
            
            gr.Markdown("### 📂 **Access Sample Detection Images**")
            gr.HTML("""
            <a href='https://www.dropbox.com/scl/fo/pvnc5nsdejvt1mhco0z8b/AI6jtnbv5s5b4aCCS-o-fxI?rlkey=181rn6o9uuyo8gqsihtqwhefj&st=qkqsyijs&dl=0' target='_blank'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/7/72/Dropbox_logo_2017.png' 
                     alt='Dropbox' width='120'>
            </a>
            """)

if __name__ == "__main__":
    demo.launch()