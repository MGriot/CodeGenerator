import base64
import zlib
import json
import os
import os
import hashlib
import qrcode
from cryptography.fernet import Fernet
from typing import Optional, Union, Tuple, Dict, Any, List
import io
from PIL import Image
from pyzbar.pyzbar import decode
import bz2
import lzma
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class QRCodeMasterProcessor:
    def __init__(
        self, encryption_key: Optional[bytes] = None, encryption_method: str = "fernet"
    ):
        """
        Initialize the master processor with optional encryption key.

        Args:
            encryption_key (Optional[bytes]): Custom encryption key.
                                              If None, a new key is generated.
            encryption_method (str): Encryption method ('fernet', 'aes')
        """
        # Generate or use provided encryption key
        if encryption_method == "aes":
            self.encryption_key = encryption_key or os.urandom(32)  # AES-256
        else:
            self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = (
            Fernet(self.encryption_key) if encryption_method == "fernet" else None
        )

    def _calculate_qr_version(self, data: str, max_version: int = 40) -> int:
        """
        Calculate the appropriate QR code version based on data length.

        Args:
            data (str): The input data to be encoded
            max_version (int): Maximum QR code version to use

        Returns:
            int: Recommended QR code version (1-40)
        """
        data_length = len(data)

        # Mapping of data lengths to QR code versions
        version_map = [
            (25, 1),
            (47, 2),
            (77, 3),
            (114, 4),
            (154, 5),
            (195, 6),
            (224, 7),
            (279, 8),
            (335, 9),
            (395, 10),
            (512, 11),
            (625, 12),
            (775, 13),
            (950, 14),
            (1145, 15),
            (1360, 16),
            (1620, 17),
            (1925, 18),
            (2250, 19),
            (2600, 20),
            (2965, 21),
            (3355, 22),
            (3775, 23),
            (4225, 24),
            (4715, 25),
            (5245, 26),
            (5815, 27),
            (6425, 28),
            (7075, 29),
            (7765, 30),
            (8495, 31),
            (9265, 32),
            (10075, 33),
            (10915, 34),
            (11795, 35),
            (12715, 36),
            (13675, 37),
            (14675, 38),
            (15715, 39),
            (16815, 40),
        ]

        for max_length, version in version_map:
            if data_length <= max_length:
                return version

        return max_version  # Maximum version if data exceeds expectations

    def compress_data(self, data: Union[str, bytes], method: str = "zlib") -> bytes:
        """
        Compress input data using the specified method.

        Args:
            data (Union[str, bytes]): Data to compress
            method (str): Compression method ('zlib', 'bz2', 'lzma')

        Returns:
            bytes: Compressed data
        """
        input_bytes = data.encode("utf-8") if isinstance(data, str) else data
        if method == "bz2":
            return bz2.compress(input_bytes)
        elif method == "lzma":
            return lzma.compress(input_bytes)
        else:
            return zlib.compress(input_bytes)

    def decompress_data(self, compressed_data: bytes, method: str = "zlib") -> bytes:
        """
        Decompress data using the specified method.

        Args:
            compressed_data (bytes): Compressed data
            method (str): Compression method ('zlib', 'bz2', 'lzma')

        Returns:
            bytes: Decompressed data
        """
        if method == "bz2":
            return bz2.decompress(compressed_data)
        elif method == "lzma":
            return lzma.decompress(compressed_data)
        else:
            return zlib.decompress(compressed_data)

    def encrypt_data(self, data: Union[str, bytes], method: str = "fernet") -> bytes:
        """
        Encrypt input data using the specified method.

        Args:
            data (Union[str, bytes]): Data to encrypt
            method (str): Encryption method ('fernet', 'aes')

        Returns:
            bytes: Encrypted data
        """
        input_bytes = data.encode("utf-8") if isinstance(data, str) else data
        if method == "aes":
            iv = os.urandom(16)
            cipher = Cipher(
                algorithms.AES(self.encryption_key[:32]),  # Ensure key size is 256 bits
                modes.CFB(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            return iv + encryptor.update(input_bytes) + encryptor.finalize()
        else:
            return self.cipher_suite.encrypt(input_bytes)

    def decrypt_data(self, encrypted_data: bytes, method: str = "fernet") -> bytes:
        """
        Decrypt data using the specified method.

        Args:
            encrypted_data (bytes): Encrypted data
            method (str): Encryption method ('fernet', 'aes')

        Returns:
            bytes: Decrypted data
        """
        if method == "aes":
            iv = encrypted_data[:16]
            cipher = Cipher(
                algorithms.AES(self.encryption_key[:32]),  # Ensure key size is 256 bits
                modes.CFB(iv),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            return decryptor.update(encrypted_data[16:]) + decryptor.finalize()
        else:
            return self.cipher_suite.decrypt(encrypted_data)

    def _encode_base64_str(self, data: bytes) -> str:
        """Convert bytes to base64 string"""
        return base64.b64encode(data).decode("utf-8")

    def _split_data_for_qr(self, data: bytes, max_chunk_size: int) -> List[bytes]:
        """Split data into chunks for QR code encoding"""
        chunks = []
        current_chunk = b""
        for i in range(0, len(data), max_chunk_size):
            current_chunk += data[i : i + max_chunk_size]
            # Estimate QR code version for the current chunk with metadata
            chunk_metadata = {
                "chunk_number": len(chunks) + 1,
                "total_chunks": 1,  # Temporarily set to 1 for estimation
                "compressed": self.compress_data(
                    current_chunk, method=self.compression_method
                )
                != current_chunk,
                "encrypted": self.encrypt_data(
                    current_chunk, method=self.encryption_method
                )
                != current_chunk,
                "compression_method": self.compression_method,
                "encryption_method": self.encryption_method,
            }
            chunk_data = json.dumps(
                {
                    "metadata": chunk_metadata,
                    "content": base64.b64encode(current_chunk).decode("utf-8"),
                }
            )
            version = self._calculate_qr_version(chunk_data)
            if version > 40:
                # If the chunk is too large, split it further
                if len(current_chunk) > max_chunk_size // 2:
                    chunks.extend(
                        self._split_data_for_qr(current_chunk, max_chunk_size // 2)
                    )
                else:
                    raise ValueError(
                        "Content too large for a single QR code. Consider improving compression or reducing chunk size."
                    )
            else:
                chunks.append(current_chunk)
                current_chunk = b""
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def generate_qr_code(
        self,
        data: Union[str, bytes],
        compress: bool = False,
        encrypt: bool = False,
        metadata: Dict[str, Any] = None,
        max_chunk_size: int = 500,  # Start with a smaller chunk size
        compression_method: str = "zlib",
        encryption_method: str = "fernet",
        split_into_chunks: bool = False,
    ) -> Tuple[List[Image.Image], List[str]]:
        if metadata is None:
            metadata = {}
        metadata["compressed"] = compress
        metadata["encrypted"] = encrypt
        metadata["compression_method"] = compression_method
        metadata["encryption_method"] = encryption_method

        # Convert input to bytes
        processed_data = data.encode("utf-8") if isinstance(data, str) else data

        # Process data based on flags
        if compress:
            processed_data = self.compress_data(
                processed_data, method=compression_method
            )

        if encrypt:
            processed_data = self.encrypt_data(processed_data, method=encryption_method)

        # Split into chunks if required
        if split_into_chunks:
            content_chunks = self._split_data_for_qr(processed_data, max_chunk_size)
        else:
            content_chunks = [processed_data]

        qr_images = []
        encoded_chunks = []

        for i, chunk_content in enumerate(content_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_number"] = i + 1
            chunk_metadata["total_chunks"] = len(content_chunks)

            # Create chunk data
            chunk_data = json.dumps(
                {
                    "metadata": chunk_metadata,
                    "content": base64.b64encode(chunk_content).decode("utf-8"),
                }
            )

            version = self._calculate_qr_version(chunk_data)
            if version > 40:
                raise ValueError(
                    "Content too large for a single QR code. Consider enabling chunking or improving compression."
                )
            elif not isinstance(version, (int, float)):
                print("Version is not a number")

            qr = qrcode.QRCode(
                version=version,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(chunk_data)
            qr.make(fit=True)

            qr_images.append(qr.make_image(fill_color="black", back_color="white"))
            encoded_chunks.append(chunk_data)

        return qr_images, encoded_chunks

    def decode_qr_data(
        self,
        qr_data: str,
        compression_method: str = "zlib",
        encryption_method: str = "fernet",
    ) -> str:
        """
        Decode data from a QR code.
        """
        try:
            # Parse the JSON data
            qr_json = json.loads(qr_data)
            metadata = qr_json["metadata"]
            content = qr_json["content"]

            # If this is a chunk, we need to parse it further
            if "chunk_number" in metadata:
                try:
                    # Parse the inner JSON content
                    inner_json = json.loads(content)
                    content = inner_json["content"]
                except json.JSONDecodeError:
                    # If not JSON, use content as is
                    pass

            # For complete data, decode base64 and process
            try:
                decoded_content = base64.b64decode(content)
            except Exception as e:
                # Add padding if needed
                padding_needed = len(content) % 4
                if padding_needed:
                    content += "=" * (4 - padding_needed)
                try:
                    decoded_content = base64.b64decode(content)
                except Exception as e2:
                    raise ValueError(f"Base64 decoding failed: {str(e2)}")

            # Process based on metadata in reverse order
            if metadata.get("encrypted", False):
                decoded_content = self.decrypt_data(
                    decoded_content,
                    method=metadata.get("encryption_method", encryption_method),
                )
            print(f"decoded_content: {decoded_content}")
            if metadata.get("compressed", False):
                decoded_content = self.decompress_data(
                    decoded_content,
                    method=metadata.get("compression_method", compression_method),
                )
            print(f"decoded_content: {decoded_content}")
            # Convert final bytes to string
            return decoded_content.decode("utf-8")

        except Exception as e:
            raise ValueError(f"Failed to decode QR data: {str(e)}")

    def reconstruct_from_qr_chunks(
        self,
        qr_chunks: List[str],
        compression_method: str = "zlib",
        encryption_method: str = "fernet",
    ) -> str:
        """
        Reconstruct the original content from multiple QR code chunks.

        Args:
            qr_chunks (List[str]): List of QR code data chunks
            compression_method (str, optional): Compression method used
            encryption_method (str, optional): Encryption method used

        Returns:
            str: Reconstructed original content
        """
        # Decode and combine chunks
        decoded_chunks = []
        for chunk in qr_chunks:
            # Decode the chunk
            decoded_chunk = self.decode_qr_data(
                chunk, compression_method, encryption_method
            )
            decoded_chunks.append(decoded_chunk)

        # Combine chunks
        combined_content = "".join(decoded_chunks)

        # Parse the combined content
        combined_json = json.loads(combined_content)
        metadata = combined_json["metadata"]
        content = combined_json["content"]

        # Decode the final content
        try:
            decoded_content = base64.b64decode(content)
        except Exception as e:
            # Add padding if needed
            padding_needed = len(content) % 4
            if padding_needed:
                content += "=" * (4 - padding_needed)
            try:
                decoded_content = base64.b64decode(content)
            except Exception as e2:
                raise ValueError(f"Base64 decoding failed: {str(e2)}")

        # Process based on metadata in reverse order
        if metadata.get("encrypted", False):
            decoded_content = self.decrypt_data(
                decoded_content,
                method=metadata.get("encryption_method", encryption_method),
            )

        if metadata.get("compressed", False):
            decoded_content = self.decompress_data(
                decoded_content,
                method=metadata.get("compression_method", compression_method),
            )

        return decoded_content.decode("utf-8")

    def save_qr_code(
        self,
        qr_images: Union[Image.Image, List[Image.Image]],
        filename: str = "qr_code.png",
    ) -> List[str]:
        """
        Save the generated QR code(s) to image file(s).

        Args:
            qr_images (Union[Image.Image, List[Image.Image]]): QR code image(s) to save
            filename (str, optional): Base output filename

        Returns:
            List[str]: List of saved filenames
        """
        if isinstance(qr_images, list):
            saved_files = []
            for i, img in enumerate(qr_images):
                # Generate filename for multiple images
                name, ext = os.path.splitext(filename)
                current_filename = f"{name}_{i+1}{ext}"
                img.save(current_filename)
                saved_files.append(current_filename)
            return saved_files
        else:
            qr_images.save(filename)
            return [filename]

    def read_qr_code_from_image(self, image_path: str) -> str:
        """
        Read the content of a QR code from an image file.

        Args:
            image_path (str): Path to the image file containing the QR code.

        Returns:
            str: Decoded content of the QR code.
        """
        try:
            # Open the image file
            image = Image.open(image_path)

            # Decode the QR code using pyzbar
            decoded_objects = decode(image)

            # Check if any QR codes are detected
            if not decoded_objects:
                raise ValueError("No QR code detected in the image.")

            # Extract the data from the first detected QR code
            qr_content = decoded_objects[0].data.decode("utf-8")

            # Return the decoded content
            return qr_content

        except Exception as e:
            raise ValueError(f"Failed to read QR code from image: {str(e)}")


class TextQRProcessor(QRCodeMasterProcessor):
    pass


class BinaryFileQRProcessor(QRCodeMasterProcessor):
    def generate_binary_file_qr_code(
        self,
        file_content: bytes,
        compress: bool = False,
        encrypt: bool = False,
        metadata: Dict[str, Any] = None,
        compression_method: str = "zlib",
        encryption_method: str = "fernet",
        split_into_chunks: bool = False,
    ) -> Tuple[List[Image.Image], List[str]]:
        """Generate a QR code from binary file content."""
        # Generate QR code from binary file content
        return self.generate_qr_code(
            file_content,
            compress,
            encrypt,
            metadata,
            compression_method=compression_method,
            encryption_method=encryption_method,
            split_into_chunks=split_into_chunks,
        )

    def reconstruct_binary_file_from_qr(
        self,
        qr_data: str,
        output_dir: Optional[str] = None,
        custom_filename: Optional[str] = None,
        compression_method: str = "zlib",
        encryption_method: str = "fernet",
    ) -> str:
        """Reconstruct a binary file from QR code data."""
        # Decode the file content
        file_content = self.decode_qr_data(
            qr_data,
            compression_method=compression_method,
            encryption_method=encryption_method,
        )

        # Prepare output path
        output_dir = output_dir or os.getcwd()
        filename = (
            os.path.splitext(custom_filename)[0]
            if custom_filename
            else "reconstructed_file"
        )
        extension = os.path.splitext(custom_filename)[1] if custom_filename else ""

        # Ensure unique filename
        full_path = os.path.join(output_dir, f"{filename}{extension}")
        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(output_dir, f"{filename}_{counter}{extension}")
            counter += 1

        # Write file content
        try:
            with open(full_path, "wb") as f:
                f.write(base64.b64decode(file_content))
        except Exception as e:
            print(f"File writing error: {str(e)}")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(str(file_content))

        return full_path


class FileQRProcessor(TextQRProcessor):

    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        encryption_method: str = "fernet",
        compression_method: str = "zlib",
    ):
        super().__init__(encryption_key, encryption_method)
        self.binary_processor = BinaryFileQRProcessor(encryption_key, encryption_method)
        self.compression_method = compression_method
        self.encryption_method = encryption_method

    def _get_file_content(self, file_path: str) -> Tuple[Union[str, bytes], bool]:
        """
        Read file content and determine how to process it.

        Args:
            file_path (str): Path to the file

        Returns:
            Tuple of (processed_content, is_text)
        """
        # Text file extensions
        text_extensions = {
            ".txt",
            ".md",
            ".py",
            ".csv",
            ".json",
            ".xml",
            ".html",
            ".css",
            ".js",
            ".log",
            ".ini",
            ".config",
        }

        # Get file extension
        file_extension = os.path.splitext(file_path)[1].lower()

        # Determine if it's a text file
        is_text = file_extension in text_extensions

        try:
            if is_text:
                # Read text files normally
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read(), True
            else:
                # Read binary files and return bytes
                with open(file_path, "rb") as f:
                    return f.read(), False
        except Exception as e:
            # Fallback for problematic files
            return str(e).encode(), False

    def generate_file_qr_code(
        self,
        file_path: str,
        compress: bool = False,
        encrypt: bool = False,
        compression_method: str = None,
        encryption_method: str = None,
        split_into_chunks: bool = False,
    ) -> Tuple[List[Image.Image], List[str]]:
        """Generate a QR code from a file's content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Use provided compression and encryption methods or default to class attributes
        compression_method = compression_method or self.compression_method
        encryption_method = encryption_method or self.encryption_method

        # Get file content and type
        file_content, is_text = self._get_file_content(file_path)

        # Calculate hash before any transformation
        if is_text:
            # For text files, hash the raw content
            file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()
        else:
            # For binary files, hash the raw bytes
            file_hash = hashlib.md5(file_content).hexdigest()

        # Generate metadata
        metadata = {
            "original_filename": os.path.basename(file_path),
            "filename": os.path.splitext(os.path.basename(file_path))[0],
            "extension": os.path.splitext(file_path)[1],
            "is_text": is_text,
            "file_size": os.path.getsize(file_path),
            "file_hash": file_hash,
        }

        # Generate QR code from file content
        if is_text:
            return self.generate_qr_code(
                file_content,
                compress,
                encrypt,
                metadata,
                compression_method=compression_method,
                encryption_method=encryption_method,
                split_into_chunks=split_into_chunks,
            )
        else:
            return self.binary_processor.generate_binary_file_qr_code(
                file_content,
                compress,
                encrypt,
                metadata,
                compression_method=compression_method,
                encryption_method=encryption_method,
                split_into_chunks=split_into_chunks,
            )

    def reconstruct_file_from_qr(
        self,
        qr_data: Union[str, List[Union[str, bytes, Image.Image]]],
        output_dir: Optional[str] = None,
        custom_filename: Optional[str] = None,
        compression_method: str = None,
        encryption_method: str = None,
    ) -> str:
        """Reconstruct a file from QR code data."""
        compression_method = compression_method or self.compression_method
        encryption_method = encryption_method or self.encryption_method

        # Handle multiple QR code chunks
        if isinstance(qr_data, list):
            # Parse each chunk to get metadata and sort by chunk number
            parsed_chunks = []
            total_chunks = None

            for chunk in qr_data:
                if isinstance(chunk, str) and chunk.startswith("{"):
                    # If it's a JSON string, parse it directly
                    chunk_json = json.loads(chunk)
                elif isinstance(chunk, str):
                    # If it's a file path, read the content and parse it
                    chunk_json = json.loads(self.read_qr_code_from_image(chunk))
                elif isinstance(chunk, bytes):
                    # If it's bytes, decode to string and parse it
                    chunk_json = json.loads(chunk.decode("utf-8"))
                elif isinstance(chunk, Image.Image):
                    # If it's an Image object, read the content and parse it
                    chunk_json = json.loads(self.read_qr_code_from_image(chunk))
                else:
                    raise ValueError(
                        "Invalid chunk type. Expected JSON string, file path, bytes, or Image object."
                    )

                metadata = chunk_json.get("metadata", {})
                chunk_number = metadata.get("chunk_number", 1)
                total_chunks = metadata.get("total_chunks",1)
                if total_chunks is None:
                    total_chunks = metadata.get("total_chunks", 1)
                parsed_chunks.append((chunk_number, chunk_json))

            # Sort chunks by chunk number
            parsed_chunks.sort(key=lambda x: x[0])

            # Verify we have all chunks
            if len(parsed_chunks) != total_chunks:
                raise ValueError(
                    f"Missing chunks. Expected {total_chunks}, got {len(parsed_chunks)}"
                )

            # Combine chunk contents
            combined_content = ""
            metadata = parsed_chunks[0][1].get(
                "metadata", {}
            )  # Use metadata from first chunk

            for _, chunk_json in parsed_chunks:
                content = chunk_json.get("content", "")
                combined_content += content

            # Create a new combined JSON structure
            combined_json = {"metadata": metadata, "content": combined_content}

            # Use the combined data for reconstruction
            file_content = self.decode_qr_data(
                json.dumps(combined_json),
                compression_method=compression_method,
                encryption_method=encryption_method,
            )
        else:
            # Parse the JSON data for single QR code
            if isinstance(qr_data, str) and qr_data.startswith("{"):
                # If it's a JSON string, parse it directly
                qr_json = json.loads(qr_data)
            elif isinstance(qr_data, str):
                # If it's a file path, read the content and parse it
                qr_json = json.loads(self.read_qr_code_from_image(qr_data))
            elif isinstance(qr_data, bytes):
                # If it's bytes, decode to string and parse it
                qr_json = json.loads(qr_data.decode("utf-8"))
            elif isinstance(qr_data, Image.Image):
                # If it's an Image object, read the content and parse it
                qr_json = json.loads(self.read_qr_code_from_image(qr_data))
            else:
                raise ValueError(
                    "Invalid QR data type. Expected JSON string, file path, bytes, or Image object."
                )

            metadata = qr_json.get("metadata", {})

            # Decode the file content
            if metadata.get("is_text", True):
                file_content = self.decode_qr_data(
                    json.dumps(qr_json),
                    compression_method=compression_method,
                    encryption_method=encryption_method,
                )
            else:
                return self.binary_processor.reconstruct_binary_file_from_qr(
                    json.dumps(qr_json),
                    output_dir,
                    custom_filename,
                    compression_method,
                    encryption_method,
                )

        # Prepare output path
        output_dir = output_dir or os.getcwd()
        filename = (
            os.path.splitext(custom_filename)[0]
            if custom_filename
            else metadata.get("filename", "reconstructed_file")
        )
        extension = (
            os.path.splitext(custom_filename)[1]
            if custom_filename
            else metadata.get("extension", "")
        )

        # Ensure unique filename
        full_path = os.path.join(output_dir, f"{filename}{extension}")
        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(output_dir, f"{filename}_{counter}{extension}")
            counter += 1

        # Verify file integrity if hash is available
        if metadata.get("file_hash"):
            try:
                if metadata.get("is_text", True):
                    current_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()
                else:
                    binary_content = base64.b64decode(file_content)
                    current_hash = hashlib.md5(binary_content).hexdigest()

                if current_hash != metadata["file_hash"]:
                    print(
                        f"Hash mismatch: Expected {metadata['file_hash']}, got {current_hash}"
                    )
                    raise ValueError("File integrity check failed")
            except Exception as e:
                print(f"Hash verification error: {str(e)}")
                raise ValueError("File integrity check failed")

        # Write file content
        try:
            if metadata.get("is_text", True):
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
            else:
                with open(full_path, "wb") as f:
                    f.write(base64.b64decode(file_content))
        except Exception as e:
            print(f"File writing error: {str(e)}")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(str(file_content))

        return full_path


class FolderQRProcessor(FileQRProcessor):
    def _traverse_folder(
        self, folder_path: str, base_path: str = None, max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Recursively traverse a folder and create a structured representation.

        Args:
            folder_path (str): Path to the folder to traverse
            base_path (str, optional): Base path for relative path calculation
            max_depth (int, optional): Maximum recursion depth

        Returns:
            Dict[str, Any]: Structured representation of the folder
        """
        # Initialize base path if not provided
        base_path = base_path or folder_path

        # Prevent excessive recursion
        if max_depth <= 0:
            return {}

        folder_structure = {
            "name": os.path.basename(folder_path),
            "path": os.relpath(folder_path, base_path),
            "type": "directory",
            "children": [],
            "metadata": {"total_files": 0, "total_size": 0},
        }

        try:
            # List all entries in the directory
            for entry in os.scandir(folder_path):
                try:
                    # Handle files
                    if entry.is_file():
                        file_info = {
                            "name": entry.name,
                            "path": os.relpath(entry.path, base_path),
                            "type": "file",
                            "size": entry.stat().st_size,
                        }

                        # Read file content (using same logic as FileQRProcessor)
                        file_content, is_text = self._get_file_content(entry.path)
                        file_info["content"] = file_content
                        file_info["is_text"] = is_text
                        # Calculate file hash
                        file_hash = hashlib.md5(
                            file_info["content"].encode("utf-8")
                        ).hexdigest()
                        file_info["hash"] = file_hash

                        folder_structure["children"].append(file_info)
                        folder_structure["metadata"]["total_files"] += 1
                        folder_structure["metadata"]["total_size"] += file_info["size"]

                    # Recursively handle subdirectories
                    elif entry.is_dir():
                        subdir = self._traverse_folder(
                            entry.path, base_path, max_depth - 1
                        )
                        if subdir:
                            folder_structure["children"].append(subdir)
                            folder_structure["metadata"]["total_files"] += subdir[
                                "metadata"
                            ]["total_files"]
                            folder_structure["metadata"]["total_size"] += subdir[
                                "metadata"
                            ]["total_size"]

                except Exception as entry_error:
                    # Log or handle individual entry errors
                    print(f"Error processing {entry.path}: {entry_error}")

        except Exception as folder_error:
            print(f"Error traversing folder {folder_path}: {folder_error}")
            return {}

        return folder_structure

    def generate_folder_qr_code(
        self,
        folder_path: str,
        compress: bool = True,
        encrypt: bool = True,
        max_depth: int = 3,
    ):
        """
        Generate a QR code representing an entire folder structure.

        Args:
            folder_path (str): Path to the folder to encode
            compress (bool, optional): Whether to compress the data
            encrypt (bool, optional): Whether to encrypt the data
            max_depth (int, optional): Maximum folder traversal depth

        Returns:
            tuple: (qr_code_image, encoded_folder_data_json)
        """
        # Validate folder exists
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Traverse the folder
        folder_structure = self._traverse_folder(folder_path, max_depth=max_depth)

        # Generate metadata
        metadata = {
            "original_foldername": os.path.basename(folder_path),
            "total_files": folder_structure["metadata"]["total_files"],
            "total_size": folder_structure["metadata"]["total_size"],
            "max_depth": max_depth,
        }

        # Convert folder structure to JSON for QR encoding
        folder_json = json.dumps(folder_structure)

        # Generate QR code
        return self.generate_qr_code(
            folder_json, compress=compress, encrypt=encrypt, metadata=metadata
        )

    def _reconstruct_entry(self, entry, current_path):
        """
        Helper function to reconstruct a single entry (file or folder) within a folder structure
        """
        if entry["type"] == "file":
            file_path = os.path.join(current_path, entry["name"])

            # Write file content
            try:
                if entry.get("is_text", True):
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(entry["content"])
                else:
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(entry["content"]))
            except Exception as e:
                print(f"Error writing file {file_path}: {e}")

        elif entry["type"] == "directory":
            subdir_path = os.path.join(current_path, entry["name"])
            os.makedirs(subdir_path)

            # Recursively reconstruct subdirectory contents
            for child in entry.get("children", []):
                self._reconstruct_entry(child, subdir_path)

    def reconstruct_folder_from_qr(
        self, qr_data: str, output_dir: str = None, restore_structure: bool = True
    ) -> str:
        """
        Reconstruct a folder from QR code data.

        Args:
            qr_data (str): Full QR code data including metadata
            output_dir (str, optional): Directory to save the reconstructed folder
            restore_structure (bool, optional): Whether to restore original folder structure

        Returns:
            str: Path to the reconstructed folder
        """
        # Decode the folder content
        folder_json = self.decode_qr_data(qr_data)
        folder_structure = json.loads(folder_json)

        # Prepare output directory
        output_dir = output_dir or os.getcwd()
        base_folder_name = folder_structure.get("name", "reconstructed_folder")
        full_output_path = os.path.join(output_dir, base_folder_name)

        # Ensure unique folder name
        counter = 1
        base_full_path = full_output_path
        while os.path.exists(full_output_path):
            full_output_path = f"{base_full_path}_{counter}"
            counter += 1

        # Create base folder
        os.makedirs(full_output_path)

        # Start reconstruction
        for child in folder_structure.get("children", []):
            self._reconstruct_entry(child, full_output_path)

        return full_output_path

    def _compress_large_data(self, data: str) -> str:
        """
        Compress large JSON data to reduce size for QR code encoding.

        Args:
            data (str): JSON string to compress

        Returns:
            str: Compressed and base64 encoded data
        """
        # Compress the data using zlib
        compressed_data = zlib.compress(data.encode("utf-8"))
        return base64.b64encode(compressed_data).decode("utf-8")

    def _chunk_folder_structure(
        self, folder_structure: Dict[str, Any], max_chunk_size: int = 10000
    ) -> List[str]:
        """
        Split large folder structure into manageable chunks.

        Args:
            folder_structure (Dict[str, Any]): Folder structure to chunk
            max_chunk_size (int, optional): Maximum size of each chunk

        Returns:
            List[str]: List of JSON chunks
        """

        def _recursive_chunk(data: Dict[str, Any]) -> List[Dict[str, Any]]:
            """
            Recursively break down the folder structure into chunks.
            """
            children = data.get("children", [])
            chunked_children = []
            current_chunk = []
            current_size = 0

            for child in children:
                child_json = json.dumps(child)
                child_size = len(child_json)

                if current_size + child_size > max_chunk_size:
                    chunked_children.append(current_chunk)
                    current_chunk = [child]
                    current_size = child_size
                else:
                    current_chunk.append(child)
                    current_size += child_size

            if current_chunk:
                chunked_children.append(current_chunk)

            return chunked_children

        # Chunk the children
        chunked_data = _recursive_chunk(folder_structure)

        # Prepare chunk metadata
        chunks = []
        for i, chunk in enumerate(chunked_data):
            chunk_data = dict(folder_structure.copy())
            chunk_data["children"] = chunk
            chunk_data["chunk_info"] = {
                "chunk_number": i + 1,
                "total_chunks": len(chunked_data),
            }
            chunks.append(json.dumps(chunk_data))

        return chunks

    def generate_folder_qr_code(
        self,
        folder_path: str,
        compress: bool = True,
        encrypt: bool = True,
        max_depth: int = 3,
        max_chunk_size: int = 10000,
    ) -> Tuple[List[Any], List[str]]:
        """
        Generate multiple QR codes for a folder structure.

        Args:
            folder_path (str): Path to the folder to encode
            compress (bool, optional): Whether to compress the data
            encrypt (bool, optional): Whether to encrypt the data
            max_depth (int, optional): Maximum folder traversal depth
            max_chunk_size (int, optional): Maximum size of each chunk

        Returns:
            tuple: (list of QR code images, list of encoded data)
        """
        # Validate folder exists
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Traverse the folder
        folder_structure = self._traverse_folder(folder_path, max_depth=max_depth)

        # Generate metadata
        metadata = {
            "original_foldername": os.path.basename(folder_path),
            "total_files": folder_structure["metadata"]["total_files"],
            "total_size": folder_structure["metadata"]["total_size"],
            "max_depth": max_depth,
        }

        # Chunk the folder structure if it's too large
        folder_chunks = self._chunk_folder_structure(folder_structure, max_chunk_size)

        # Generate QR codes for each chunk
        qr_images = []
        encoded_chunks = []

        for chunk in folder_chunks:
            # Optionally compress the chunk
            if compress:
                chunk = self._compress_large_data(chunk)

            # Generate QR code for the chunk
            qr_image, encoded_chunk = self.generate_qr_code(
                chunk,
                compress=False,  # Already compressed if needed
                encrypt=encrypt,
                metadata=metadata,
            )

            qr_images.append(qr_image)
            encoded_chunks.append(encoded_chunk)

        return qr_images, encoded_chunks

    def reconstruct_folder_from_qr_chunks(
        self, qr_chunks: List[str], output_dir: str = None
    ) -> str:
        """
        Reconstruct a folder from multiple QR code chunks.

        Args:
            qr_chunks (List[str]): List of QR code data chunks
            output_dir (str, optional): Directory to save the reconstructed folder

        Returns:
            str: Path to the reconstructed folder
        """
        # Decode and combine chunks
        decoded_chunks = []
        for chunk in qr_chunks:
            # Decode the chunk
            decoded_chunk = json.loads(self.decode_qr_data(chunk))
            decoded_chunks.append(decoded_chunk)

        # Sort chunks if chunk information is available
        if decoded_chunks and "chunk_info" in decoded_chunks[0]:
            decoded_chunks.sort(
                key=lambda x: x.get("chunk_info", {}).get("chunk_number", 0)
            )

        # Combine children from all chunks
        base_structure = decoded_chunks[0]
        combined_children = []

        for chunk in decoded_chunks:
            combined_children.extend(chunk.get("children", []))

        base_structure["children"] = combined_children

        # Use existing reconstruction method
        folder_json = json.dumps(base_structure)
        return self.reconstruct_folder_from_qr(
            self.generate_qr_code(folder_json)[1], output_dir
        )


# Example usage demonstrating all processors
def ensure_output_dir():
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def test_text_qr():
    output_dir = ensure_output_dir()
    text_processor = TextQRProcessor()
    original_text = "Hello, this is a test message for QR code generation!"

    # Test different compression and encryption methods
    methods = [
        ("zlib", "fernet"),
        ("bz2", "fernet"),
        ("lzma", "fernet"),
        ("zlib", "aes"),
        ("bz2", "aes"),
        ("lzma", "aes"),
    ]

    for compression_method, encryption_method in methods:
        print(
            f"Testing with compression: {compression_method}, encryption: {encryption_method}"
        )
        text_processor = TextQRProcessor(encryption_method=encryption_method)
        text_qr_images, text_encoded_data = text_processor.generate_qr_code(
            original_text,
            compress=True,
            encrypt=True,
            compression_method=compression_method,
            encryption_method=encryption_method,
        )
        saved_files = text_processor.save_qr_code(
            text_qr_images,
            os.path.join(
                output_dir, f"text_qr_code_{compression_method}_{encryption_method}.png"
            ),
        )
        print(f"Saved QR codes to: {', '.join(saved_files)}")
        decoded_text = text_processor.decode_qr_data(
            text_encoded_data[0],
            compression_method=compression_method,
            encryption_method=encryption_method,
        )
        print("Text Decoding Test:")
        print("Original Text:", original_text)
        print("Decoded Text:", decoded_text)
        print("Texts Match:", original_text == decoded_text)
        print()


def test_file_qr():
    print("=" * 10)
    print("File test")
    output_dir = ensure_output_dir()
    file_processor = FileQRProcessor()
    test_file_path = "large_test_file.txt"

    # Test different compression and encryption methods
    methods = [
        ("zlib", "fernet"),
        ("bz2", "fernet"),
        ("lzma", "fernet"),
        ("zlib", "aes"),
        ("bz2", "aes"),
        ("lzma", "aes"),
    ]

    for compression_method, encryption_method in methods:
        print(
            f"Testing with compression: {compression_method}, encryption: {encryption_method}"
        )
        try:
            file_qr_images, file_encoded_data = file_processor.generate_file_qr_code(
                test_file_path,
                compress=True,
                encrypt=True,
                compression_method=compression_method,
                encryption_method=encryption_method,
                split_into_chunks=True,
            )
            saved_files = file_processor.save_qr_code(
                file_qr_images,
                os.path.join(
                    output_dir,
                    f"file_qr_code_{compression_method}_{encryption_method}.png",
                ),
            )
            print(f"Saved file QR codes to: {', '.join(saved_files)}")

            # Read QR codes back from files
            #qr_data_list = read_qr_codes_from_files(saved_files)

            # Reconstruct the file
            reconstructed_file_path = file_processor.reconstruct_file_from_qr(
                saved_files
            )
            print("File Reconstruction Test:")
            print("Original File:", test_file_path)
            print("Reconstructed File:", reconstructed_file_path)

            # Compare file contents
            with open(test_file_path, "r") as f:
                original_content = f.read()

            with open(reconstructed_file_path, "r") as f:
                reconstructed_content = f.read()

            print("Files Match:", original_content == reconstructed_content)
            print()
        except ValueError as e:
            print(f"Error: {e}. Consider enabling chunking or improving compression.")
            print()


def test_folder_qr():
    output_dir = ensure_output_dir()
    folder_processor = FolderQRProcessor()
    test_folder = "test"

    # Generate QR codes for the folder
    folder_qr_images, folder_encoded_chunks = folder_processor.generate_folder_qr_code(
        test_folder, compress=True, encrypt=True, max_depth=2  # Limit depth to 2 levels
    )

    # Save QR codes
    for i, qr_image in enumerate(folder_qr_images):
        folder_processor.save_qr_code(
            qr_image, os.path.join(output_dir, f"folder_qr_code_{i}.png")
        )

    # Reconstruct folder
    reconstructed_folder = folder_processor.reconstruct_folder_from_qr_chunks(
        folder_encoded_chunks, output_dir=""
    )

    print(f"Folder reconstructed at: {reconstructed_folder}")
    print()

    # Create a folder with files for testing chunked encoding
    test_folder_chunk = "test_chunk"
    os.makedirs(test_folder_chunk, exist_ok=True)

    # Create some files in the folder
    with open(os.path.join(test_folder_chunk, "file1.txt"), "w") as f:
        f.write("This is a test file 1.")
    with open(os.path.join(test_folder_chunk, "file2.txt"), "w") as f:
        f.write("This is a test file 2.")

    # Generate chunked QR codes
    qr_images, encoded_chunks = folder_processor.generate_folder_qr_code(
        test_folder_chunk, compress=True, encrypt=True, max_depth=2, max_chunk_size=500
    )

    # Save the qr codes to files
    for i, qr_image in enumerate(qr_images):
        folder_processor.save_qr_code(
            qr_image, os.path.join(output_dir, f"folder_qr_code_chunk_{i}.png")
        )

    # Reconstruct folder from chunks
    reconstructed_folder_chunk = folder_processor.reconstruct_folder_from_qr_chunks(
        encoded_chunks, output_dir="."
    )

    print(f"Chunked folder reconstructed at: {reconstructed_folder_chunk}")

def test_read_qr_code_from_image(path:str) -> str:
    processor = QRCodeMasterProcessor()
    qr_content = processor.read_qr_code_from_image(path)
    print(qr_content)

def main():
    test_read_qr_code_from_image(r"C:\Users\Admin\Documents\Coding\CodeGenerator\input\file_qr_code.png")
    # test_text_qr()
    test_file_qr()
    # test_folder_qr()


if __name__ == "__main__":
    main()
