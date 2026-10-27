"""Generate training_pb2.py and training_pb2_grpc.py beside training.proto."""
from pathlib import Path
import grpc_tools.protoc


ROOT = Path(__file__).resolve().parent


def main() -> None:
    code = grpc_tools.protoc.main(
        ["grpc_tools.protoc", f"-I{ROOT}", f"--python_out={ROOT}", f"--grpc_python_out={ROOT}", str(ROOT / "training.proto")]
    )
    if code:
        raise SystemExit(code)
    generated = ROOT / "training_pb2_grpc.py"
    text = generated.read_text(encoding="utf-8")
    text = text.replace("from . from . import training_pb2 as training__pb2", "from . import training_pb2 as training__pb2")
    text = text.replace("import training_pb2 as training__pb2", "from . import training_pb2 as training__pb2")
    generated.write_text(text, encoding="utf-8")
    grpc_file = ROOT / "training_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    text = text.replace("import training_pb2 as training__pb2", "from . import training_pb2 as training__pb2")
    grpc_file.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
