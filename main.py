import os
import base64
import zlib
import json
import qrcode
import hashlib
import secrets
from typing import Union, Optional, List, Tuple
from pathlib import Path
from time import time

from pylibdmtx.pylibdmtx import encode as encode_datamatrix
import barcode
from barcode.writer import ImageWriter
from PIL import Image
import cv2
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class DocumentCodeManager:
    def __init__(self, seed: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the DocumentCodeManager with advanced encoding capabilities.

        :param seed: Optional seed for generating encryption key
        :param output_dir: Directory to save generated image files
        """
        self.seed = seed or self._generate_random_seed()
        self.encryption_key = self._generate_key_from_seed(self.seed)
        self.cipher_suite = Fernet(self.encryption_key)

        # Set up output directory
        self.output_dir = output_dir or os.path.join(os.getcwd(), "generated_codes")
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_random_seed(self, length: int = 32) -> str:
        """
        Generate a random seed for encryption.

        :param length: Length of the seed
        :return: Random seed string
        """
        return secrets.token_hex(length // 2)

    def _generate_key_from_seed(self, seed: str) -> bytes:
        """
        Generate a cryptographic key from a seed using PBKDF2.

        :param seed: Seed string for key generation
        :return: Cryptographic key
        """
        salt = b"document_code_salt"  # Consistent salt for reproducibility
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(seed.encode()))

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt the input data using Fernet encryption.

        :param data: Input data to encrypt
        :return: Encrypted bytes
        """
        # Ensure data is in bytes
        if isinstance(data, str):
            data = data.encode("utf-8")

        return self.cipher_suite.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt the input data using Fernet encryption.

        :param encrypted_data: Encrypted bytes
        :return: Decrypted string
        """
        return self.cipher_suite.decrypt(encrypted_data).decode("utf-8")

    def compress_and_encode(self, input_path: Union[str, Path]) -> str:
        """
        Compress and encode input (text, file, or folder) into a compact representation.

        :param input_path: Path to text, file, or folder
        :return: Compressed and encrypted data string
        """
        input_path = Path(input_path)

        # Determine input type and process accordingly
        if input_path.is_file():
            return self._process_file(input_path)
        elif input_path.is_dir():
            return self._process_folder(input_path)
        else:
            return self._process_text(str(input_path))

    def _process_text(self, text: str) -> str:
        """
        Process and compress text input.

        :param text: Input text
        :return: Compressed and encrypted text representation
        """
        # Compress text
        compressed_text = self._advanced_compress(text)

        # Encrypt compressed text
        encrypted_data = base64.b64encode(self.encrypt_data(compressed_text)).decode(
            "utf-8"
        )

        return encrypted_data

    def _process_file(self, file_path: Path) -> str:
        """
        Process and compress a single file.

        :param file_path: Path to the file
        :return: Compressed and encrypted file representation
        """
        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Create metadata dictionary
        file_metadata = {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size": file_path.stat().st_size,
            "content_hash": hashlib.md5(file_content).hexdigest(),
        }

        # Combine metadata and content, then compress
        combined_data = json.dumps(
            {
                "metadata": file_metadata,
                "content": base64.b64encode(file_content).decode("utf-8"),
            }
        )

        compressed_data = self._advanced_compress(combined_data)

        # Encrypt compressed data
        encrypted_data = base64.b64encode(self.encrypt_data(compressed_data)).decode(
            "utf-8"
        )

        return encrypted_data

    def _process_folder(self, folder_path: Path) -> str:
        """
        Process and compress an entire folder with its contents.

        :param folder_path: Path to the folder
        :return: Compressed and encrypted folder representation
        """
        folder_contents = []

        # Recursively process files in the folder
        for item in folder_path.rglob("*"):
            if item.is_file():
                with open(item, "rb") as f:
                    file_content = f.read()

                # Create file metadata
                file_metadata = {
                    "relative_path": str(item.relative_to(folder_path)),
                    "filename": item.name,
                    "extension": item.suffix,
                    "size": item.stat().st_size,
                    "content_hash": hashlib.md5(file_content).hexdigest(),
                    "content": base64.b64encode(file_content).decode("utf-8"),
                }

                folder_contents.append(file_metadata)

        # Create folder metadata
        folder_metadata = {
            "folder_name": folder_path.name,
            "total_files": len(folder_contents),
            "contents": folder_contents,
        }

        # Convert to JSON, compress, and_encrypt
        folder_json = json.dumps(folder_metadata)
        compressed_data = self._advanced_compress(folder_json)
        encrypted_data = base64.b64encode(self.encrypt_data(compressed_data)).decode(
            "utf-8"
        )

        return encrypted_data

    def _advanced_compress(self, data: str) -> bytes:
        """
        Advanced compression using multiple techniques.

        :param data: Input data to compress
        :return: Compressed bytes
        """
        # Convert to bytes
        data_bytes = data.encode("utf-8")

        # First level compression
        compressed = zlib.compress(data_bytes, level=9)

        return compressed

    def decode_data(self, encoded_data: str) -> Union[str, dict]:
        """
        Decode and decompress data from various sources.

        :param encoded_data: Encoded data string
        :return: Decoded and decompressed data
        """
        try:
            # Decrypt base64 encoded data
            decrypted_data = self.decrypt_data(base64.b64decode(encoded_data))

            # Decompress data
            decompressed_data = self._advanced_decompress(
                decrypted_data.encode("utf-8")
            )

            # Try parsing as JSON (for file/folder data)
            try:
                return json.loads(decompressed_data)
            except json.JSONDecodeError:
                # If not JSON, return as plain text
                return decompressed_data

        except Exception as e:
            return f"Decoding error: {str(e)}"

    def _advanced_decompress(self, compressed_data: bytes) -> str:
        """
        Advanced decompression method.

        :param compressed_data: Compressed bytes
        :return: Decompressed string
        """
        # Decompress using zlib
        decompressed = zlib.decompress(compressed_data)

        return decompressed.decode("utf-8")

    def reconstruct_folder(
        self, encoded_data: str, output_path: Optional[str] = None
    ) -> str:
        """
        Reconstruct a folder from encoded data.

        :param encoded_data: Encoded folder data
        :param output_path: Optional output path for reconstruction
        :return: Path to reconstructed folder
        """
        # Decode the data
        decoded_folder_data = self.decode_data(encoded_data)

        if not isinstance(decoded_folder_data, dict):
            raise ValueError("Invalid folder data")

        # Determine output path
        if output_path is None:
            output_path = os.path.join(
                self.output_dir, decoded_folder_data["folder_name"]
            )

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Reconstruct files
        for file_info in decoded_folder_data["contents"]:
            # Determine full file path
            file_relative_path = file_info["relative_path"]
            full_file_path = os.path.join(output_path, file_relative_path)

            # Create directory for file if it doesn't exist
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

            # Decode and write file content
            file_content = base64.b64decode(file_info["content"])

            # Verify content hash
            content_hash = hashlib.md5(file_content).hexdigest()
            if content_hash != file_info["content_hash"]:
                print(f"Warning: Hash mismatch for file {file_relative_path}")

            # Write file
            with open(full_file_path, "wb") as f:
                f.write(file_content)

        return output_path

    def generate_code_from_input(
        self, input_path: Union[str, Path], code_type: str, filename: str
    ) -> Image.Image:
        """
        Generate a QR code or DataMatrix code from input (text, file, or folder).

        :param input_path: Path to text, file, or folder
        :param code_type: Type of code to generate ('qr' or 'datamatrix')
        :param filename: Output filename
        :return: Generated code image
        """
        start_time = time()
        print(f"Compressing and encoding input path: {input_path}")
        encoded_data = self.compress_and_encode(input_path)
        print(f"Encoded data length: {len(encoded_data)}")

        if code_type == "qr":
            print("Creating QR code...")
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(encoded_data)
            qr.make(fit=True)

            print("Saving QR code...")
            filepath = os.path.join(self.output_dir, filename)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(filepath)
            print(f"QR Code saved to: {filepath}")
        elif code_type == "datamatrix":
            print("Creating DataMatrix code...")
            encoded_bytes = encoded_data.encode("utf-8")
            datamatrix = encode_datamatrix(encoded_bytes)
            img = Image.fromarray(datamatrix)
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            print(f"DataMatrix code saved to: {filepath}")
        else:
            raise ValueError("Invalid code type. Use 'qr' or 'datamatrix'.")

        end_time = time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        return img

    def read_and_reconstruct_code(
        self, image_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Read a QR code or DataMatrix code and reconstruct its contents.

        :param image_path: Path to the code image
        :param output_path: Optional output path for reconstruction
        :return: Path to reconstructed folder or decoded data
        """
        # Ensure the image path is correct
        image_path = os.path.join(self.output_dir, image_path)

        # Read code
        if image_path.endswith(".png"):
            detector = cv2.QRCodeDetector()
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            data, bbox, _ = detector.detectAndDecode(img)
        elif image_path.endswith(".datamatrix"):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            decoded = pylibdmtx.decode(img)
            if decoded:
                data = decoded[0].data.decode("utf-8")
            else:
                data = None
        else:
            raise ValueError(
                "Unsupported image format. Use .png for QR codes or .datamatrix for DataMatrix codes."
            )

        if data:
            # Attempt to decode and reconstruct
            try:
                # Try to reconstruct folder
                return self.reconstruct_folder(data, output_path)
            except Exception:
                # If not a folder, return decoded data
                return self.decode_data(data)

        return "No code found"


def main():
    # Create DocumentCodeManager instance
    doc_manager = DocumentCodeManager(seed="folder_reconstruction")

    # Example text
    example_text = "Hello, World!"

    # Example text file
    example_text_file_path = (
        r"C:\Users\Admin\Documents\Coding\DocumentCodeManager\test\test1.txt"
    )
    with open(example_text_file_path, "w") as f:
        f.write("This is an example text file.\nIt contains some sample text.\n")

    # Generate QR code from text
    print("Generating QR code from text...")
    qr_code_text_path = "text_qr_code.png"
    doc_manager.generate_code_from_input(example_text, "qr", qr_code_text_path)
    print(f"QR Code saved to: {qr_code_text_path}")

    # Generate QR code from text file
    print("Generating QR code from text file...")
    qr_code_file_path = "text_file_qr_code.png"
    doc_manager.generate_code_from_input(
        example_text_file_path, "qr", qr_code_file_path
    )
    print(f"QR Code saved to: {qr_code_file_path}")

    # Read and reconstruct QR code from text
    print("Reading and reconstructing QR code from text...")
    reconstructed_text = doc_manager.read_and_reconstruct_code(qr_code_text_path)
    print(f"Reconstructed text: {reconstructed_text}")

    # Read and reconstruct QR code from text file
    print("Reading and reconstructing QR code from text file...")
    reconstructed_file_path = doc_manager.read_and_reconstruct_code(qr_code_file_path)
    print(f"Reconstructed file saved to: {reconstructed_file_path}")

    # Verify the reconstructed file content
    with open(reconstructed_file_path, "r") as f:
        reconstructed_file_content = f.read()
    print(f"Reconstructed file content: {reconstructed_file_content}")


if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          