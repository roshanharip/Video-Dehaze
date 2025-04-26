import cv2
import torch
import numpy as np
from models import FFA


class VideoDehazer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = FFA(gps=3, blocks=19).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.scaler = torch.cuda.amp.GradScaler()

    def dehaze_frame(self, frame):
        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inference with FP16
        with torch.no_grad(), torch.cuda.amp.autocast():
            dehazed = self.model(frame)

        # Postprocess
        dehazed = dehazed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        dehazed = dehazed.astype(np.uint8)
        return cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)


def main():
    # Initialize dehazer
    dehazer = VideoDehazer("ots_train_ffa_3_19.pk", device="cuda")

    # Open webcam (use "input.mp4" for video files)
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input FPS: {fps}")

    # Optional: Write output video
    # out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed (adjust as needed)
        frame = cv2.resize(frame, (640, 480))

        # Dehaze frame
        dehazed_frame = dehazer.dehaze_frame(frame)

        # Display
        cv2.imshow("Original", frame)
        cv2.imshow("Dehazed", dehazed_frame)

        # Write to output (uncomment if needed)
        # out.write(dehazed_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()