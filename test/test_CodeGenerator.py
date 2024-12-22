import os
import sys
import json
import shutil
import pytest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.CodeGenerator import (
    QRCodeMasterProcessor,
    TextQRProcessor,
    FileQRProcessor,
    FolderQRProcessor,
)

@pytest.fixture
def setup_test_files():
    # Create test directory and files
    os.makedirs("test_folder", exist_ok=True)
    with open("test_file.txt", "w") as f:
        f.write("Test content")
    with open(os.path.join("test_folder", "test1.txt"), "w") as f:
        f.write("Test content 1")
    
    yield
    
    # Cleanup
    if os.path.exists("test_file.txt"):
        os.remove("test_file.txt")
    if os.path.exists("test_folder"):
        shutil.rmtree("test_folder")

class TestQRCodeMasterProcessor:
    def test_init(self):
        processor = QRCodeMasterProcessor()
        assert processor is not None
        assert processor.encryption_key is not None

    def test_encode_decode(self):
        processor = QRCodeMasterProcessor()
        test_data = "Test string"
        qr_image, encoded_data = processor.generate_qr_code(test_data)
        assert qr_image is not None
        decoded = processor.decode_qr_data(encoded_data)
        assert decoded == test_data

class TestTextQRProcessor:
    def test_init(self):
        processor = TextQRProcessor()
        assert processor is not None

    def test_generate_qr_code(self):
        processor = TextQRProcessor()
        text = "Hello, World!"
        qr_images, encoded_data = processor.generate_qr_code(text)
        assert len(qr_images) > 0
        assert len(encoded_data) > 0

    def test_decode_qr_data(self):
        processor = TextQRProcessor()
        test_data = "Test data"
        _, encoded_data = processor.generate_qr_code(test_data)
        decoded_text = processor.decode_qr_data(encoded_data[0])
        assert decoded_text == test_data

class TestFileQRProcessor:
    def test_init(self):
        processor = FileQRProcessor()
        assert processor is not None

    def test_generate_file_qr_code(self, setup_test_files):
        processor = FileQRProcessor()
        qr_images, encoded_data = processor.generate_file_qr_code("test_file.txt")
        assert len(qr_images) > 0
        assert len(encoded_data) > 0

    def test_reconstruct_file_from_qr(self, setup_test_files):
        processor = FileQRProcessor()
        _, encoded_data = processor.generate_file_qr_code("test_file.txt")
        reconstructed_path = processor.reconstruct_file_from_qr(encoded_data[0])
        assert os.path.exists(reconstructed_path)
        with open(reconstructed_path, 'r') as f:
            assert f.read() == "Test content"

class TestFolderQRProcessor:
    def test_init(self):
        processor = FolderQRProcessor()
        assert processor is not None

    def test_generate_folder_qr_code(self, setup_test_files):
        processor = FolderQRProcessor()
        qr_images, encoded_data = processor.generate_folder_qr_code("test_folder")
        assert len(qr_images) > 0
        assert len(encoded_data) > 0

    def test_reconstruct_folder_from_qr(self, setup_test_files):
        processor = FolderQRProcessor()
        _, encoded_data = processor.generate_folder_qr_code("test_folder")
        reconstructed_path = processor.reconstruct_folder_from_qr(encoded_data[0])
        assert os.path.exists(reconstructed_path)
        assert os.path.exists(os.path.join(reconstructed_path, "test1.txt"))
