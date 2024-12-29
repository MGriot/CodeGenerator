import base64
import zlib
import json
import os
import hashlib
import qrcode
import lzma
import bz2
import binascii
from enum import Enum
from dataclasses import dataclass
from cryptography.fernet import Fernet
from typing import Optional, Union, Tuple, Dict, Any, List
from PIL import Image

import logging

logging.basicConfig(level=logging.DEBUG)

class CompressionType(Enum):
    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    BZ2 = "bz2"

class CompressionLevel(Enum):
    NONE = 0
    FAST = 1
    BALANCED = 6
    BEST = 9

@dataclass
class CompressionConfig:
    type: CompressionType = CompressionType.ZLIB
    level: CompressionLevel = CompressionLevel.BALANCED

class QRCodeMasterProcessor:
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize the master processor with optional encryption key.

        Args:
            encryption_key (Optional[bytes]): Custom encryption key.
                                              If None, a new key is generated.
        """
        # Generate or use provided encryption key
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def _calculate_qr_version(self, data: str, max_version: int = 40) -> int:
        """Calculate the appropriate QR code version based on data length."""
        data_length = len(data)

        # Version map with more accurate capacity estimates
        version_map = [
            (17, 1),    # Version 1
            (32, 2),    # Version 2
            (53, 3),    # Version 3
            (78, 4),    # Version 4
            (106, 5),   # Version 5
            (134, 6),   # Version 6
            (154, 7),   # Version 7
            (192, 8),   # Version 8
            (230, 9),   # Version 9
            (271, 10),  # Version 10
            (321, 11),
            (367, 12),
            (425, 13),
            (458, 14),
            (520, 15),
            (586, 16),
            (644, 17),
            (718, 18),
            (792, 19),
            (858, 20),
            (929, 21),
            (1003, 22),
            (1091, 23),
            (1171, 24),
            (1273, 25),
            (1367, 26),
            (1465, 27),
            (1528, 28),
            (1628, 29),
            (1732, 30),
            (1840, 31),
            (1952, 32),
            (2068, 33),
            (2188, 34),
            (2303, 35),
            (2431, 36),
            (2563, 37),
            (2699, 38),
            (2809, 39),
            (2953, 40), # Version 40
        ]

        for max_length, version in version_map:
            if data_length <= max_length:
                return version

        return max_version  # Maximum version if data exceeds expectations

    def compress_data(self, data: Union[str, bytes], config: Optional[CompressionConfig] = None) -> bytes:
        """Compress data using specified algorithm and level."""
        if config is None:
            config = CompressionConfig()

        input_bytes = data.encode("utf-8") if isinstance(data, str) else data

        if config.type == CompressionType.NONE:
            logging.debug(f"Compression type: NONE")
            return input_bytes

        try:
            if config.type == CompressionType.ZLIB:
                compressed_data = zlib.compress(input_bytes, level=config.level.value)
            elif config.type == CompressionType.LZMA:
                compressed_data = lzma.compress(input_bytes, preset=config.level.value)
            elif config.type == CompressionType.BZ2:
                compressed_data = bz2.compress(input_bytes, compresslevel=config.level.value)
            logging.debug(f"Compressed data: {compressed_data}")
            return compressed_data
        except Exception as e:
            logging.error(f"Compression failed with {config.type.value}: {str(e)}")
            return zlib.compress(input_bytes, level=CompressionLevel.BALANCED.value)

    def decompress_data(self, compressed_data: bytes, compression_type: str) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.NONE.value:
            logging.debug(f"Decompression type: NONE")
            return compressed_data

        try:
            if compression_type == CompressionType.ZLIB.value:
                decompressed_data = zlib.decompress(compressed_data)
            elif compression_type == CompressionType.LZMA.value:
                decompressed_data = lzma.decompress(compressed_data)
            elif compression_type == CompressionType.BZ2.value:
                decompressed_data = bz2.decompress(compressed_data)
            logging.debug(f"Decompressed data: {decompressed_data}")
            return decompressed_data
        except Exception as e:
            logging.error(f"Decompression failed with {compression_type}: {str(e)}")
            raise ValueError(f"Decompression failed with {compression_type}: {str(e)}")

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt input data using Fernet symmetric encryption.

        Args:
            data (Union[str, bytes]): Data to encrypt

        Returns:
            bytes: Encrypted data
        """
        input_bytes = data.encode("utf-8") if isinstance(data, str) else data
        encrypted_data = self.cipher_suite.encrypt(input_bytes)
        logging.debug(f"Encrypted data: {encrypted_data}")
        return encrypted_data


    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet symmetric encryption."""
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            logging.debug(f"Decrypted data: {decrypted_data}")
            return decrypted_data
        except Exception as e:
            logging.error(f"Decryption failed: {str(e)}")
            raise ValueError(f"Decryption failed: {str(e)}")

    def _encode_base64_str(self, data: bytes) -> str:
        """Convert bytes to base64 string"""
        return base64.b64encode(data).decode('utf-8')

    def _split_data_for_qr(self, data: str, max_chunk_size: int) -> List[str]:
        """Split data into chunks for QR code encoding"""
        return [
            data[i: i + max_chunk_size] for i in range(0, len(data), max_chunk_size)
        ]

    def _estimate_qr_capacity(self, version: int, error_correction: qrcode.constants.ERROR_CORRECT_H) -> int:
        """Estimate QR code capacity for a given version."""
        # Conservative capacity estimates for error correction level H
        base_capacity = (((version * 4 + 17) ** 2) // 8) * 0.3  # 30% of max capacity due to error correction
        return int(base_capacity)

    def _suggest_compression_method(self, data_size: int) -> str:
        """Suggest the best compression method based on data size."""
        suggestions = []
        if data_size > 2953:  # Max QR v40 capacity
            suggestions.append("Content is too large for a single QR code.")
            suggestions.append("Consider the following options:")
            suggestions.append("1. Use LZMA compression for best compression ratio")
            suggestions.append("2. Split content into multiple QR codes")
            suggestions.append(f"3. Current data size: {data_size} bytes")
            suggestions.append(f"4. Maximum QR code capacity: ~2953 bytes")
            num_qrs = (data_size // 2000) + 1
            suggestions.append(f"5. Recommended: Split into {num_qrs} QR codes")
        return "\n".join(suggestions)

    def generate_qr_code(
        self,
        data: Union[str, bytes],
        compress: bool = False,
        compression_config: Optional[CompressionConfig] = None,
        encrypt: bool = False,
        metadata: Dict[str, Any] = None,
        max_chunk_size: int = 1000
    ) -> Tuple[List[Image.Image], List[str]]:
        """Generate QR code with automatic chunking for large content."""
        if metadata is None:
            metadata = {}

        try:
            # Process data first to check final size
            processed_data = data.encode("utf-8") if isinstance(data, str) else data

            # Try compression if enabled
            if compress and compression_config:
                processed_data = self.compress_data(processed_data, compression_config)
                metadata["compressed"] = True
                metadata["compression_type"] = compression_config.type.value

            # Apply encryption if enabled
            if encrypt:
                processed_data = self.encrypt_data(processed_data)
                metadata["encrypted"] = True

            # Convert to base64
            base64_data = base64.b64encode(processed_data).decode('utf-8')

            # Create initial data structure
            qr_data = {"metadata": metadata, "content": base64_data}
            full_data = json.dumps(qr_data)

            # Check if data is too large
            data_size = len(full_data)
            logging.debug(f"Full data size: {data_size} bytes")
            logging.debug(f"Estimated QR capacity: {self._estimate_qr_capacity(40, qrcode.constants.ERROR_CORRECT_H)} bytes")
            if data_size > self._estimate_qr_capacity(40, qrcode.constants.ERROR_CORRECT_H):
                suggestions = self._suggest_compression_method(data_size)
                if not compress:
                    raise ValueError(f"Data too large for QR code. Try enabling compression.\n{suggestions}")
                else:
                    # Automatically split into multiple chunks
                    chunk_size = self._estimate_qr_capacity(30, qrcode.constants.ERROR_CORRECT_H)
                    chunks = self._split_data_for_qr(full_data, chunk_size)
                    metadata["auto_chunked"] = True
                    metadata["total_chunks"] = len(chunks)
            else:
                chunks = [full_data]

            # Generate QR codes
            qr_images = []
            encoded_chunks = []

            for i, chunk_content in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_number"] = i + 1
                chunk_metadata["total_chunks"] = len(chunks)

                chunk_data = json.dumps({
                    "metadata": chunk_metadata,
                    "content": chunk_content
                })

                # Create QR code with appropriate version
                version = self._calculate_qr_version(chunk_data)
                logging.debug(f"QR code version: {version}")
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

            if len(chunks) > 1:
                logging.debug(f"Content split into {len(chunks)} QR codes due to size.")
                logging.debug(f"Use all QR codes in sequence for complete data.")

            return qr_images, encoded_chunks

        except Exception as e:
            if "Invalid version" in str(e):
                suggestions = self._suggest_compression_method(len(full_data))
                raise ValueError(f"Content too large for QR code.\n{suggestions}")
            raise ValueError(f"Failed to generate QR code: {str(e)}")
    def decode_qr_data(self, qr_data: str) -> Union[str, bytes]:
        """Decode QR data, handling both text and binary content."""
        try:
            qr_json = json.loads(qr_data)
            metadata = qr_json.get("metadata", {})
            content = qr_json.get("content", "")
            is_text = metadata.get("is_text", True)

            padding_attempts = [
                content,
                content + "=" * ((4 - len(content) % 4) % 4),
                content.rstrip("="),
                content.rstrip("=") + "=",
                content.rstrip("=") + "=="
            ]

            decoded_content = None
            for attempt in padding_attempts:
                try:
                    decoded_content = base64.b64decode(attempt)
                    break
                except binascii.Error:
                    continue

            if decoded_content is None:
                raise ValueError("Base64 decoding failed")

            if metadata.get("encrypted", False):
                try:
                    decoded_content = self.decrypt_data(decoded_content)
                except ValueError:
                    # Skip decryption if it fails
                    pass

            if metadata.get("compressed", False):
                compression_type = CompressionType(metadata.get("compression_type"))
                decoded_content = self.decompress_data(decoded_content, compression_type.value)

            # Return string for text files, bytes for binary
            if is_text:
                try:
                    return decoded_content.decode("utf-8")
                except UnicodeDecodeError:
                    return decoded_content
            return decoded_content

        except Exception as e:
            logging.error(f"Failed to decode QR data: {str(e)}")
            raise ValueError(f"Failed to decode QR data: {str(e)}")

    def save_qr_code(
        self, qr_images: Union[Image.Image, List[Image.Image]], filename: str = "qr_code.png"
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


class TextQRProcessor(QRCodeMasterProcessor):
    def generate_text_qr(
        self,
        text: str,
        compression_type: CompressionType = CompressionType.ZLIB,
        compression_level: CompressionLevel = CompressionLevel.BALANCED,
        encrypt: bool = False
    ) -> Tuple[List[Image.Image], List[str]]:
        """User-friendly method to generate QR code with specified compression."""
        compression_config = CompressionConfig(compression_type, compression_level)
        return self.generate_qr_code(
            text,
            compress=compression_type != CompressionType.NONE,
            compression_config=compression_config,
            encrypt=encrypt
        )


class FileQRProcessor(QRCodeMasterProcessor):

    def _get_file_content(self, file_path: str) -> Tuple[str, bool, str]:
        """Read file content and determine how to process it."""
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

        file_extension = os.path.splitext(file_path)[1].lower()
        is_text = file_extension in text_extensions

        try:
            if is_text:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            else:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                    file_content = base64.b64encode(file_bytes).decode("utf-8")
        except Exception as e:
            file_content = base64.b64encode(str(e).encode()).decode("utf-8")
            is_text = False

        # Calculate hash before any transformation
        if is_text:
            file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()
        else:
            file_hash = hashlib.md5(base64.b64decode(file_content)).hexdigest()

        logging.debug(f"Original file hash: {file_hash}")
        return file_content, is_text, file_hash

    def generate_file_qr(
        self,
        file_path: str,
        compression_type: CompressionType = CompressionType.ZLIB,
        compression_level: CompressionLevel = CompressionLevel.BALANCED,
        encrypt: bool = False
    ) -> Tuple[List[Image.Image], List[str]]:
        """User-friendly method to generate QR code with specified compression."""
        compression_config = CompressionConfig(compression_type, compression_level)
        return self.generate_file_qr_code(
            file_path,
            compress=True,
            compression_config=compression_config,
            encrypt=encrypt
        )

    def generate_file_qr_code(
        self,
        file_path: str,
        compress: bool = False,
        compression_config: Optional[CompressionConfig] = None,
        encrypt: bool = False
    ) -> Tuple[List[Image.Image], List[str]]:
        """Generate a QR code from a file's content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file content and type
        file_content, is_text, file_hash = self._get_file_content(file_path)

        # Generate metadata
        metadata = {
            "original_filename": os.path.basename(file_path),
            "filename": os.path.splitext(os.path.basename(file_path))[0],
            "extension": os.path.splitext(file_path)[1],
            "is_text": is_text,
            "file_size": os.path.getsize(file_path),
            "file_hash": file_hash
        }

        # Initialize scaled_file_content with the original file content
        scaled_file_content = file_content

        # Try compression if enabled
        if compress and compression_config:
            scaled_file_content = self.compress_data(scaled_file_content, compression_config)
            metadata["compressed"] = True
            metadata["compression_type"] = compression_config.type.value
            logging.debug(f"Compressed file content: {scaled_file_content}")

        # Apply encryption if enabled
        if encrypt:
            scaled_file_content = self.encrypt_data(scaled_file_content)
            metadata["encrypted"] = True
            logging.debug(f"Encrypted file content: {scaled_file_content}")

        # Convert to base64
        base64_data = base64.b64encode(scaled_file_content).decode('utf-8')
        logging.debug(f"Base64 encoded file content: {base64_data}")

        qr_data = {"metadata": metadata, "content": base64_data}
        full_data = json.dumps(qr_data)
        logging.debug(f"Full QR data: {full_data}")

        data_size = len(full_data)
        logging.debug(f"Full data size: {data_size} bytes")
        logging.debug(f"Estimated QR capacity: {self._estimate_qr_capacity(40, qrcode.constants.ERROR_CORRECT_H)} bytes")

        if data_size > self._estimate_qr_capacity(40, qrcode.constants.ERROR_CORRECT_H):
            suggestions = self._suggest_compression_method(data_size)
            if not compress:
                raise ValueError(f"Data too large for QR code. Try enabling compression.\n{suggestions}")
            else:
                # Automatically split into multiple chunks
                chunk_size = self._estimate_qr_capacity(30, qrcode.constants.ERROR_CORRECT_H)
                chunks = self._split_data_for_qr(full_data, chunk_size)
                metadata["auto_chunked"] = True
                metadata["total_chunks"] = len(chunks)
        else:
            chunks = [full_data]

        qr_images = []
        encoded_chunks = []

        for i, chunk_content in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_number"] = i + 1
            chunk_metadata["total_chunks"] = len(chunks)
            
            chunk_data = json.dumps({
                "metadata": chunk_metadata,
                "content": chunk_content
            })
            logging.debug(f"Chunk data: {chunk_data}")

            version = self._calculate_qr_version(chunk_data)
            logging.debug(f"QR code version: {version}")
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

        if len(chunks) > 1:
            logging.debug(f"Content split into {len(chunks)} QR codes due to size.")
            logging.debug(f"Use all QR codes in sequence for complete data.")

        return qr_images, encoded_chunks
    def reconstruct_file_from_qr(
        self,
        qr_data: str,
        output_dir: Optional[str] = None,
        custom_filename: Optional[str] = None,
    ) -> str:
        """Reconstruct a file from QR code data."""
        # Parse the JSON data
        try:
            qr_json = json.loads(qr_data)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse QR data: {str(e)}")
            raise ValueError(f"Failed to parse QR data: {str(e)}")

        metadata = qr_json.get("metadata", {})
        chunk_content = qr_json.get("content", "")

        # Decode the file content
        file_content = self.decode_qr_data(chunk_content)
        logging.debug(f"Decoded file content: {file_content}")

        # Prepare output path
        output_dir = output_dir or os.getcwd()
        filename = (os.path.splitext(custom_filename)[0] if custom_filename 
                    else metadata.get("filename", "reconstructed_file"))
        extension = (os.path.splitext(custom_filename)[1] if custom_filename 
                    else metadata.get("extension", ""))
        
        # Ensure unique filename
        full_path = os.path.join(output_dir, f"{filename}{extension}")
        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(output_dir, f"{filename}_{counter}{extension}")
            counter += 1

        # Verify file integrity
        if metadata.get("file_hash"):
            try:
                if metadata.get("is_text", True):
                    # For text files, verify using the decoded content
                    # Ensure file_content is bytes for hashing
                    if isinstance(file_content, str):
                        current_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()
                    else:
                        current_hash = hashlib.md5(file_content).hexdigest()
                else:
                    # For binary files, verify using the decoded bytes
                    binary_content = file_content
                    current_hash = hashlib.md5(binary_content).hexdigest()

                logging.debug(f"Reconstructed file hash: {current_hash}")

                if current_hash != metadata["file_hash"]:
                    raise ValueError(f"File integrity check failed\nHash mismatch: Expected {metadata['file_hash']}, got {current_hash}")
            except Exception as e:
                logging.error(f"Hash verification error: {str(e)}")
                raise ValueError("File integrity check failed")

        # Write file content
        try:
            if metadata.get("is_text", True):
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
            else:
                with open(full_path, "wb") as f:
                    f.write(file_content)
        except Exception as e:
            logging.error(f"File writing error: {str(e)}")
            raise ValueError("Failed to write reconstructed file")

        return full_path

    def reconstruct_file_from_qr_chunks(
        self, qr_chunks: List[str], output_dir: str = None
    ) -> str:
        """Reconstruct a file from multiple QR code chunks."""
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
        combined_data = ""
        for chunk in decoded_chunks:
            combined_data += chunk.get("content", "")

        # Use existing reconstruction method
        return self.reconstruct_file_from_qr(combined_data, output_dir)

class FolderQRProcessor(FileQRProcessor):
    def _traverse_folder(
        self, folder_path: str, base_path: str = None, max_depth: int = 3
    ) -> Dict[str, Any]:
        """Recursively traverse a folder and create a structured representation."""
        # Initialize base path if not provided
        base_path = base_path or folder_path

        # Prevent excessive recursion
        if max_depth <= 0:
            return {
                "name": os.path.basename(folder_path),
                "path": os.path.relpath(folder_path, base_path),
                "type": "directory",
                "children": [],
                "metadata": {"total_files": 0, "total_size": 0},
            }

        folder_structure = {
            "name": os.path.basename(folder_path),
            "path": os.path.relpath(folder_path, base_path),
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
                            "path": os.path.relpath(entry.path, base_path),
                            "type": "file",
                            "size": entry.stat().st_size,
                        }

                        # Read file content
                        file_content, is_text = self._get_file_content(entry.path)
                        file_info["content"] = file_content
                        file_info["is_text"] = is_text
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
                        if subdir:  # Changed condition
                            folder_structure["children"].append(subdir)
                            folder_structure["metadata"]["total_files"] += subdir["metadata"]["total_files"]
                            folder_structure["metadata"]["total_size"] += subdir["metadata"]["total_size"]

                except Exception as entry_error:
                    print(f"Error processing {entry.path}: {entry_error}")
                    continue  # Skip problematic entries but continue processing

        except Exception as folder_error:
            print(f"Error traversing folder {folder_path}: {folder_error}")

        return folder_structure  # Always return the structure, even if empty

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

            qr_images.extend(qr_image)
            encoded_chunks.extend(encoded_chunk)

        return qr_images, encoded_chunks


    def reconstruct_folder_from_qr_chunks(
        self, qr_chunks: List[str], output_dir: str = None
    ) -> str:
        """Reconstruct a folder from multiple QR code chunks."""
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
        return self.reconstruct_folder_from_qr(folder_json, output_dir)


# Example usage demonstrating all processors
def test_compression_string(
    text: str, filename_prefix: str = "text"
) -> Dict[str, Dict[str, int]]:
    """Test all compression types and levels for text."""
    text_processor = TextQRProcessor()
    results = {
        comp_type.value: {level.value: 0 for level in CompressionLevel}
        for comp_type in CompressionType
    }

    # Test without compression first
    try:
        qr_images, encoded_data = text_processor.generate_qr_code(
            text, compress=False, encrypt=False  # Changed to False to reduce complexity
        )
        results[CompressionType.NONE.value][CompressionLevel.NONE.value] = len(
            encoded_data[0]
        )

        # Save the baseline QR code
        text_processor.save_qr_code(qr_images, f"{filename_prefix}_none_baseline.png")

        # Verify decoding
        decoded_text = text_processor.decode_qr_data(encoded_data[0])
        if decoded_text == text:
            print(f"Baseline test successful: {len(encoded_data[0])} bytes")
        else:
            print("Baseline decode verification failed")

    except Exception as e:
        print(f"Baseline test failed: {str(e)}")

    # Test each compression type and level
    for comp_type in [CompressionType.ZLIB, CompressionType.LZMA, CompressionType.BZ2]:
        for comp_level in CompressionLevel:
            # Skip invalid combinations
            if comp_type == CompressionType.BZ2 and comp_level == CompressionLevel.NONE:
                continue

            try:
                print(
                    f"\nTesting {comp_type.value} compression at level {comp_level.value}"
                )

                # Generate QR code with compression
                qr_images, encoded_data = text_processor.generate_text_qr(
                    text,
                    compression_type=comp_type,
                    compression_level=comp_level,
                    encrypt=False,  # Changed to False to reduce complexity
                )

                # Save QR codes
                saved_files = text_processor.save_qr_code(
                    qr_images,
                    f"{filename_prefix}_{comp_type.value}_{comp_level.value}.png",
                )

                # Verify decoding
                decoded_text = text_processor.decode_qr_data(encoded_data[0])
                success = decoded_text == text

                # Store results
                results[comp_type.value][comp_level.value] = len(encoded_data[0])

                print(f"Success: {success}")
                print(f"Size: {len(encoded_data[0])} bytes")
                print(f"Files: {', '.join(saved_files)}")

            except Exception as e:
                print(
                    f"Error with {comp_type.value} level {comp_level.value}: {str(e)}"
                )

    return results


def test_compression_file(
    file_path: str, filename_prefix: str = "file"
) -> Dict[str, Dict[str, int]]:
    """Test all compression types and levels for file."""
    file_processor = FileQRProcessor()
    results = {}

    # Read original file content for verification
    with open(file_path, "r", encoding="utf-8") as f:
        original_content = f.read()

    for comp_type in CompressionType:
        results[comp_type.value] = {}
        for comp_level in CompressionLevel:
            print(
                f"\nTesting {comp_type.value} compression at level {comp_level.value}"
            )

            # Generate QR code
            qr_images, encoded_data = file_processor.generate_file_qr(
                file_path,
                compression_type=comp_type,
                compression_level=comp_level,
                encrypt=True,
            )

            # Save QR codes
            saved_files = file_processor.save_qr_code(
                qr_images, f"{filename_prefix}_{comp_type.value}_{comp_level.value}.png"
            )

            # Reconstruct and verify
            reconstructed_file = file_processor.reconstruct_file_from_qr(
                encoded_data[0],
                custom_filename=f"reconstructed_{comp_type.value}_{comp_level.value}.txt",
            )

            with open(reconstructed_file, "r", encoding="utf-8") as f:
                reconstructed_content = f.read()

            success = reconstructed_content == original_content
            qr_size = len(encoded_data[0])

            results[comp_type.value][comp_level.value] = qr_size

            print(f"Compression: {comp_type.value}")
            print(f"Level: {comp_level.value}")
            print(f"Success: {success}")
            print(f"QR Data Size: {qr_size} bytes")
            print(f"Files: {', '.join(saved_files)}")
            print(f"Reconstructed: {reconstructed_file}")

            # Clean up reconstructed file
            os.remove(reconstructed_file)

    return results


def compare_compression_results(results: Dict[str, Dict[str, int]]):
    """Compare and display compression results."""
    print("\nCompression Results Summary:")
    print("=" * 60)
    print(f"{'Type':<10} {'Level':<8} {'Size (bytes)':<12} {'Reduction %':<12}")
    print("-" * 60)

    # Find baseline size - use the first successful result if none compression failed
    baseline = results[CompressionType.NONE.value].get(CompressionLevel.NONE.value, 0)
    if baseline == 0:
        # Find first non-zero size as baseline
        for comp_type in results:
            for level, size in results[comp_type].items():
                if size > 0:
                    baseline = size
                    break
            if baseline > 0:
                break

    if baseline == 0:
        print("Error: No successful compression results found")
        return

    # Display results
    for comp_type, levels in results.items():
        for level, size in levels.items():
            if size > 0:  # Only show successful results
                reduction = ((baseline - size) / baseline) * 100
                print(f"{comp_type:<10} {level:<8} {size:<12} {reduction:>6.2f}%")


def string():
    """Test string compression"""
    print("\nTesting String Compression")
    print("=" * 50)
    
    # Test with smaller repetitive content
    test_text = "Hello, this is a test message! " * 10  # Reduced repetitions
    results = test_compression_string(test_text)
    compare_compression_results(results)

def file():
    """Test file compression"""
    print("\nTesting File Compression")
    print("=" * 50)
    
    # Create a test file with repetitive content
    test_file_path = "test_file.txt"
    test_content = "This is a test file content with some repetitive text. " * 20
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        results = test_compression_file(test_file_path)
        compare_compression_results(results)
    finally:
        # Clean up
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    string()
    file()
    #folder()
