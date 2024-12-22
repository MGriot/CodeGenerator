import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.CodeGenerator import (
    QRCodeMasterProcessor,
    TextQRProcessor,
    FileQRProcessor,
    FolderQRProcessor,
)


class TestQRcodeMasterProcessor:
    def test_init(self):
        processor = QRcodeMasterProcessor()
        assert processor is not None

    def test_process(self):
        processor = QRcodeMasterProcessor()
        # Mock input and output files
        input_file = "input.txt"
        output_file = "output.txt"
        processor.process(input_file, output_file)
        assert True  # Replace with actual assertion


class TestTextQRProcessor:
    def test_init(self):
        processor = TextQRProcessor()
        assert processor is not None

    def test_generate_qr_code(self):
        processor = TextQRProcessor()
        text = "Hello, World!"
        qr_image, encoded_data = processor.generate_qr_code(text)
        assert qr_image is not None
        assert encoded_data is not None

    def test_decode_qr_data(self):
        processor = TextQRProcessor()
        encoded_data = " encoded data "
        decoded_text = processor.decode_qr_data(encoded_data)
        assert decoded_text is not None


class TestFileQRProcessor:
    def test_init(self):
        processor = FileQRProcessor()
        assert processor is not None

    def test_generate_file_qr_code(self):
        processor = FileQRProcessor()
        file_path = "test_file.txt"
        qr_image, encoded_data = processor.generate_file_qr_code(file_path)
        assert qr_image is not None
        assert encoded_data is not None

    def test_reconstruct_file_from_qr(self):
        processor = FileQRProcessor()
        encoded_data = " encoded data "
        reconstructed_file_path = processor.reconstruct_file_from_qr(encoded_data)
        assert reconstructed_file_path is not None


class TestFolderQRProcessor:
    def test_init(self):
        processor = FolderQRProcessor()
        assert processor is not None

    def test_generate_folder_qr_code(self):
        processor = FolderQRProcessor()
        folder_path = "test_folder"
        qr_image, encoded_data = processor.generate_folder_qr_code(folder_path)
        assert qr_image is not None
        assert encoded_data is not None

    def test_reconstruct_folder_from_qr(self):
        processor = FolderQRProcessor()
        encoded_data = " encoded data "
        reconstructed_folder_path = processor.reconstruct_folder_from_qr(encoded_data)
        assert reconstructed_folder_path is not None
