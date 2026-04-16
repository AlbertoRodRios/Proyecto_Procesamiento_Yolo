from ultralytics import YOLO

def main() -> None:
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="data/pipeline_A/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="results/detect_experiments",
        name="pipeline_A_e50_gpu_seed123",
        workers=4,
        device=0,
        pretrained=True,
        seed=123,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()