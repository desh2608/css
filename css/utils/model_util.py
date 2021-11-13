import onnxruntime
import torch
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_onnx_model(onnx_model_path, sess_options, device):
    if device == torch.device("cuda") and onnxruntime.get_device() != "GPU":
        logging.warning("To use GPU, please install onnxruntime-gpu")
        device = torch.device("cpu")
    model = onnxruntime.InferenceSession(onnx_model_path, sess_options=sess_options)
    return model, device


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )
