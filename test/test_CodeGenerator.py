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

import unittest
from PIL import Image
import json
import base64


class TestQRCodeProcessors(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextQRProcessor()
        self.file_processor = FileQRProcessor()
        self.folder_processor = FolderQRProcessor()
        self.test_file_path = "test_file.txt"
        self.test_folder = "test_folder"
        self.test_folder_chunk = "test_folder_chunk"
        self.test_text = "This is a test string."
        self.test_file_content = "This is a test file content."
        self.test_folder_structure = {
            "name": "test_folder",
            "path": ".",
            "type": "directory",
            "children": [
                {
                    "name": "test_file1.txt",
                    "path": "test_file1.txt",
                    "type": "file",
                    "size": 25,
                    "content": "This is test file 1.",
                    "is_text": True,
                    "hash": "59748444444444444444444444444444",
                },
                {
                    "name": "test_file2.txt",
                    "path": "test_file2.txt",
                    "type": "file",
                    "size": 25,
                    "content": "This is test file 2.",
                    "is_text": True,
                    "hash": "59748444444444444444444444444444",
                },
            ],
            "metadata": {"total_files": 2, "total_size": 50},
        }

        # Create test file
        with open(self.test_file_path, "w") as f:
            f.write(self.test_file_content)

        # Create test folder and files
        os.makedirs(self.test_folder, exist_ok=True)
        with open(os.path.join(self.test_folder, "test_file1.txt"), "w") as f:
            f.write("This is test file 1.")
        with open(os.path.join(self.test_folder, "test_file2.txt"), "w") as f:
            f.write("This is test file 2.")

        # Create test folder for chunked encoding
        os.makedirs(self.test_folder_chunk, exist_ok=True)
        with open(os.path.join(self.test_folder_chunk, "file1.txt"), "w") as f:
            f.write("This is a test file 1.")
        with open(os.path.join(self.test_folder_chunk, "file2.txt"), "w") as f:
            f.write("This is a test file 2.")

    def tearDown(self):
        # Clean up test files and folders
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)
        if os.path.exists(self.test_folder_chunk):
            shutil.rmtree(self.test_folder_chunk)

    def test_qr_code_master_processor_init(self):
        processor = QRCodeMasterProcessor()
        self.assertIsNotNone(processor.encryption_key)

    def test_calculate_qr_version(self):
        self.assertEqual(self.text_processor._calculate_qr_version("A" * 20), 1)
        self.assertEqual(self.text_processor._calculate_qr_version("A" * 50), 2)
        self.assertEqual(self.text_processor._calculate_qr_version("A" * 1000), 10)

    def test_compress_and_decompress_data(self):
        compressed_data = self.text_processor.compress_data(self.test_text)
        decompressed_data = self.text_processor.decompress_data(compressed_data)
        self.assertEqual(decompressed_data.decode("utf-8"), self.test_text)

    def test_encrypt_and_decrypt_data(self):
        encrypted_data = self.text_processor.encrypt_data(self.test_text)
        decrypted_data = self.text_processor.decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data.decode("utf-8"), self.test_text)

    def test_text_qr_code_generation_and_decoding(self):
        qr_images, encoded_data = self.text_processor.generate_qr_code(
            self.test_text, compress=True, encrypt=True
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_data, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_data))
        decoded_text = self.text_processor.decode_qr_data(encoded_data[0])
        self.assertEqual(decoded_text, self.test_text)

    def test_file_qr_code_generation_and_reconstruction(self):
        qr_images, encoded_data = self.file_processor.generate_file_qr_code(
            self.test_file_path, compress=True, encrypt=True
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_data, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_data))
        reconstructed_file = self.file_processor.reconstruct_file_from_qr(
            encoded_data[0]
        )
        with open(reconstructed_file, "r") as f:
            reconstructed_content = f.read()
        self.assertEqual(reconstructed_content, self.test_file_content)
        os.remove(reconstructed_file)

    def test_folder_qr_code_generation_and_reconstruction(self):
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            self.test_folder, compress=True, encrypt=True, max_depth=2
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_chunks, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_chunks))
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir="."
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        self.assertTrue(
            os.path.exists(os.path.join(reconstructed_folder, "test_file1.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(reconstructed_folder, "test_file2.txt"))
        )
        shutil.rmtree(reconstructed_folder)

    def test_chunked_folder_qr_code_generation_and_reconstruction(self):
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            self.test_folder_chunk,
            compress=True,
            encrypt=True,
            max_depth=2,
            max_chunk_size=500,
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_chunks, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_chunks))
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir="."
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        self.assertTrue(
            os.path.exists(os.path.join(reconstructed_folder, "file1.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(reconstructed_folder, "file2.txt"))
        )
        shutil.rmtree(reconstructed_folder)

    def test_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            self.file_processor.generate_file_qr_code("non_existent_file.txt")

    def test_folder_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            self.folder_processor.generate_folder_qr_code("non_existent_folder")

    def test_reconstruct_file_with_custom_filename(self):
        qr_images, encoded_data = self.file_processor.generate_file_qr_code(
            self.test_file_path, compress=True, encrypt=True
        )
        reconstructed_file = self.file_processor.reconstruct_file_from_qr(
            encoded_data[0], custom_filename="custom_file.txt"
        )
        self.assertTrue(os.path.exists(reconstructed_file))
        self.assertTrue("custom_file.txt" in reconstructed_file)
        os.remove(reconstructed_file)

    def test_reconstruct_file_with_output_dir(self):
        qr_images, encoded_data = self.file_processor.generate_file_qr_code(
            self.test_file_path, compress=True, encrypt=True
        )
        output_dir = "output_test_dir"
        os.makedirs(output_dir, exist_ok=True)
        reconstructed_file = self.file_processor.reconstruct_file_from_qr(
            encoded_data[0], output_dir=output_dir
        )
        self.assertTrue(os.path.exists(reconstructed_file))
        self.assertTrue(output_dir in reconstructed_file)
        os.remove(reconstructed_file)
        shutil.rmtree(output_dir)

    def test_reconstruct_folder_with_output_dir(self):
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            self.test_folder, compress=True, encrypt=True, max_depth=2
        )
        output_dir = "output_test_dir"
        os.makedirs(output_dir, exist_ok=True)
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir=output_dir
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        self.assertTrue(output_dir in reconstructed_folder)
        shutil.rmtree(reconstructed_folder)
        shutil.rmtree(output_dir)

    def test_reconstruct_folder_with_custom_name(self):
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            self.test_folder, compress=True, encrypt=True, max_depth=2
        )
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir=".",
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        self.assertTrue("test_folder" in reconstructed_folder)
        shutil.rmtree(reconstructed_folder)

    def test_reconstruct_folder_with_chunked_output_dir(self):
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            self.test_folder_chunk,
            compress=True,
            encrypt=True,
            max_depth=2,
            max_chunk_size=500,
        )
        output_dir = "output_test_dir"
        os.makedirs(output_dir, exist_ok=True)
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir=output_dir
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        self.assertTrue(output_dir in reconstructed_folder)
        shutil.rmtree(reconstructed_folder)
        shutil.rmtree(output_dir)

    def test_reconstruct_folder_with_chunked_custom_name(self):
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            self.test_folder_chunk,
            compress=True,
            encrypt=True,
            max_depth=2,
            max_chunk_size=500,
        )
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir="."
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        self.assertTrue("test_folder_chunk" in reconstructed_folder)
        shutil.rmtree(reconstructed_folder)

    def test_large_text_qr_code_generation(self):
        large_text = "A" * 2000
        qr_images, encoded_data = self.text_processor.generate_qr_code(
            large_text, compress=True, encrypt=True
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_data, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_data))
        decoded_text = self.text_processor.decode_qr_data(encoded_data[0])
        self.assertEqual(decoded_text, large_text)

    def test_large_file_qr_code_generation(self):
        large_file_content = "B" * 2000
        with open("large_test_file.txt", "w") as f:
            f.write(large_file_content)
        qr_images, encoded_data = self.file_processor.generate_file_qr_code(
            "large_test_file.txt", compress=True, encrypt=True
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_data, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_data))
        reconstructed_file = self.file_processor.reconstruct_file_from_qr(
            encoded_data[0]
        )
        with open(reconstructed_file, "r") as f:
            reconstructed_content = f.read()
        self.assertEqual(reconstructed_content, large_file_content)
        os.remove("large_test_file.txt")
        os.remove(reconstructed_file)

    def test_large_folder_qr_code_generation(self):
        large_folder = "large_test_folder"
        os.makedirs(large_folder, exist_ok=True)
        for i in range(3):
            with open(os.path.join(large_folder, f"file_{i}.txt"), "w") as f:
                f.write("C" * 1000)
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            large_folder, compress=True, encrypt=True, max_depth=2
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_chunks, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_chunks))
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir="."
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        shutil.rmtree(large_folder)
        shutil.rmtree(reconstructed_folder)

    def test_large_chunked_folder_qr_code_generation(self):
        large_folder = "large_test_folder_chunk"
        os.makedirs(large_folder, exist_ok=True)
        for i in range(3):
            with open(os.path.join(large_folder, f"file_{i}.txt"), "w") as f:
                f.write("D" * 1000)
        qr_images, encoded_chunks = self.folder_processor.generate_folder_qr_code(
            large_folder,
            compress=True,
            encrypt=True,
            max_depth=2,
            max_chunk_size=1000,
        )
        self.assertIsInstance(qr_images, list)
        self.assertTrue(all(isinstance(img, Image.Image) for img in qr_images))
        self.assertIsInstance(encoded_chunks, list)
        self.assertTrue(all(isinstance(data, str) for data in encoded_chunks))
        reconstructed_folder = self.folder_processor.reconstruct_folder_from_qr_chunks(
            encoded_chunks, output_dir="."
        )
        self.assertTrue(os.path.exists(reconstructed_folder))
        shutil.rmtree(large_folder)
        shutil.rmtree(reconstructed_folder)

if __name__ == "__main__":
    unittest.main()
