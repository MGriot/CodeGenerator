import base64
import zlib
import json
import os
import hashlib
import qrcode
from cryptography.fernet import Fernet
from typing import Optional, Union, Tuple, Dict, Any, List
import io
from PIL import Image


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

    def _calculate_qr_version(self, data: str, max_version: int = 10) -> int:
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

    def compress_data(self, data: Union[str, bytes]) -> bytes:
        """
        Compress input data using zlib.

        Args:
            data (Union[str, bytes]): Data to compress

        Returns:
            bytes: Compressed data
        """
        # Convert to bytes if input is string
        input_bytes = data.encode("utf-8") if isinstance(data, str) else data
        return zlib.compress(input_bytes)

    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        Decompress data using zlib.

        Args:
            compressed_data (bytes): Compressed data

        Returns:
            bytes: Decompressed data
        """
        return zlib.decompress(compressed_data)

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt input data using Fernet symmetric encryption.

        Args:
            data (Union[str, bytes]): Data to encrypt

        Returns:
            bytes: Encrypted data
        """
        # Convert to bytes if input is string
        input_bytes = data.encode("utf-8") if isinstance(data, str) else data
        return self.cipher_suite.encrypt(input_bytes)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using Fernet symmetric encryption.

        Args:
            encrypted_data (bytes): Encrypted data

        Returns:
            bytes: Decrypted data
        """
        return self.cipher_suite.decrypt(encrypted_data)

    def generate_qr_code(
        self,
        data: Union[str, bytes],
        compress: bool = False,
        encrypt: bool = False,
        metadata: Dict[str, Any] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Generate a QR code from input data with optional compression and encryption.

        Args:
            data (Union[str, bytes]): Data to encode
            compress (bool, optional): Whether to compress the data
            encrypt (bool, optional): Whether to encrypt the data
            metadata (Dict[str, Any], optional): Additional metadata to include in the QR code

        Returns:
            tuple: (qr_code_image, encoded_data_json)
        """
        # Prepare metadata for tracking processing steps
        if metadata is None:
            metadata = {}
        metadata["compressed"] = compress
        metadata["encrypted"] = encrypt

        # Convert input to bytes if it's a string
        processed_data = data if not isinstance(data, str) else data.encode("utf-8")

        # Process data based on compression and encryption flags
        if compress:
            processed_data = self.compress_data(processed_data)
            processed_data = base64.b64encode(processed_data).decode("utf-8")

        if encrypt:
            processed_data = self.encrypt_data(processed_data)
            processed_data = base64.b64encode(processed_data).decode("utf-8")

        # Combine metadata and processed data
        qr_data = json.dumps({"metadata": metadata, "content": processed_data})

        # Determine QR code version dynamically
        version = self._calculate_qr_version(qr_data)

        # Create QR code with adaptive sizing
        qr = qrcode.QRCode(
            version=version,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        # Create image
        img = qr.make_image(fill_color="black", back_color="white")

        return img, qr_data

    def decode_qr_data(self, qr_data: str) -> str:
        """
        Decode data from a QR code, automatically detecting and applying
        decompression and decryption based on metadata.

        Args:
            qr_data (str): Full QR code data including metadata

        Returns:
            str: Decoded data
        """
        # Parse the JSON data
        qr_json = json.loads(qr_data)
        metadata = qr_json["metadata"]
        qr_content = qr_json["content"]

        # Process data based on metadata
        decoded_data = qr_content.encode("utf-8")

        if metadata.get("encrypted", False):
            decoded_data = self.decrypt_data(base64.b64decode(decoded_data))

        if metadata.get("compressed", False):
            decoded_data = self.decompress_data(base64.b64decode(decoded_data))

        return decoded_data.decode("utf-8")

    def save_qr_code(
        self, qr_image: Image.Image, filename: str = "qr_code.png"
    ) -> None:
        """
        Save the generated QR code to an image file.

        Args:
            qr_image (Image.Image): QR code image to save
            filename (str, optional): Output filename
        """
        qr_image.save(filename)


class TextQRProcessor(QRCodeMasterProcessor):
    def decode_qr_text(self, qr_data: str) -> str:
        """
        Decode text from a QR code.

        Args:
            qr_data (str): Full QR code data including metadata

        Returns:
            str: Decoded text
        """
        decoded_bytes = self.decode_qr_data(qr_data)
        return decoded_bytes


class FileQRProcessor(QRCodeMasterProcessor):

    def _get_file_content(self, file_path: str) -> Tuple[str, bool]:
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
                # Read binary files and convert to base64
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                    return base64.b64encode(file_bytes).decode("utf-8"), False
        except Exception as e:
            # Fallback for problematic files
            return base64.b64encode(str(e).encode()).decode("utf-8"), False

    def generate_file_qr_code(
        self, file_path: str, compress: bool = False, encrypt: bool = False
    ) -> Tuple[Image.Image, str]:
        """
        Generate a QR code from a file's content.

        Args:
            file_path (str): Path to the file
            compress (bool, optional): Whether to compress the file content
            encrypt (bool, optional): Whether to encrypt the file content

        Returns:
            tuple: (qr_code_image, encoded_file_data_json)
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file content and type
        file_content, is_text = self._get_file_content(file_path)

        # Generate comprehensive metadata
        metadata = {
            "original_filename": os.path.basename(file_path),
            "filename": os.path.splitext(os.path.basename(file_path))[0],
            "extension": os.path.splitext(file_path)[1],
            "is_text": is_text,
            "file_size": os.path.getsize(file_path),
        }

        # Calculate file hash
        file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()
        metadata["file_hash"] = file_hash

        # Generate QR code from file content
        return self.generate_qr_code(file_content, compress, encrypt, metadata)

    def reconstruct_file_from_qr(
        self,
        qr_data: str,
        output_dir: Optional[str] = None,
        custom_filename: Optional[str] = None,
    ) -> str:
        """
        Reconstruct a file from QR code data.

        Args:
            qr_data (str): Full QR code data including metadata
            output_dir (Optional[str]): Directory to save the file
            custom_filename (Optional[str]): Custom filename to use

        Returns:
            str: Path to the reconstructed file
        """
        # Parse the JSON data
        qr_json = json.loads(qr_data)
        metadata = qr_json.get("metadata", {})
        print(metadata)

        # Decode the file content
        file_content = self.decode_qr_data(qr_data)

        # Prepare output directory
        output_dir = output_dir or os.getcwd()

        # Determine filename
        if custom_filename:
            filename = os.path.splitext(custom_filename)[0]
            extension = os.path.splitext(custom_filename)[1] or metadata.get(
                "extension", ""
            )
        else:
            filename = metadata.get("filename", "reconstructed_file")
            extension = metadata.get("extension", "")
        # Construct full path
        full_path = os.path.join(output_dir, f"{filename}{extension}")

        # Ensure unique filename
        counter = 1
        base_full_path = full_path
        while os.path.exists(full_path):
            filename_part = os.path.splitext(base_full_path)[0]
            ext_part = os.path.splitext(base_full_path)[1]
            full_path = f"{filename_part}_{counter}{ext_part}"
            counter += 1

        # Verify file integrity
        if metadata.get("file_hash"):
            current_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()
            if current_hash != metadata.get("file_hash"):
                raise ValueError("File integrity check failed")

        # Write file
        try:
            if metadata.get("is_text", True):
                # Write text files
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
            else:
                # Decode and write binary files
                with open(full_path, "wb") as f:
                    f.write(base64.b64decode(file_content))
        except Exception as e:
            # Fallback writing mechanism
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(str(file_content))

        return full_path

    def _pad_base64(self, base64_str: str) -> str:
        """
        Ensure the base64 string is properly padded.

        Args:
            base64_str (str): Base64-encoded string

        Returns:
            str: Properly padded base64 string
        """
        padding_needed = len(base64_str) % 4
        if padding_needed:
            base64_str += "=" * (4 - padding_needed)
        return base64_str


class FolderQRProcessor(FileQRProcessor):
    def _estimate_qr_code_capacity(self, version: int) -> int:
        """
        Estimate the maximum data capacity for a given QR code version.
        """
        capacity_map = {
            1: 25,
            2: 47,
            3: 77,
            4: 114,
            5: 154,
            6: 195,
            7: 224,
            8: 279,
            9: 335,
            10: 395,
            11: 512,
            12: 625,
            13: 775,
            14: 950,
            15: 1145,
            16: 1360,
            17: 1620,
            18: 1925,
            19: 2250,
            20: 2600,
            21: 2965,
            22: 3355,
            23: 3775,
            24: 4225,
            25: 4715,
            26: 5245,
            27: 5815,
            28: 6425,
            29: 7075,
            30: 7765,
            31: 8495,
            32: 9265,
            33: 10075,
            34: 10915,
            35: 11795,
            36: 12715,
            37: 13675,
            38: 14675,
            39: 15715,
            40: 16815,
        }
        return capacity_map.get(
            version, 25
        )  # Default to smallest if version is out of range

    def _smart_chunk_folder_structure(
        self, folder_structure: Dict[str, Any], max_version: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Intelligently chunk the folder structure to fit within QR code limitations.
        """
        print(f"DEBUG: Starting smart chunking for folder structure")
        print(
            f"DEBUG: Total children to chunk: {len(folder_structure.get('children', []))}"
        )
        print(f"DEBUG: Using max QR code version: {max_version}")

        # Start with the full folder structure
        base_structure = folder_structure.copy()
        base_structure["children"] = []

        # Track chunks and remaining children
        chunks = []
        remaining_children = folder_structure.get("children", [])

        while remaining_children:
            # Create a chunk
            current_chunk = base_structure.copy()
            current_chunk["children"] = []
            current_chunk_size = 0

            # Add children to the chunk
            while (
                remaining_children
                and current_chunk_size < self._estimate_qr_code_capacity(max_version)
            ):
                # Take the next child
                child = remaining_children.pop(0)

                # Estimate child size
                child_json = json.dumps(child)
                child_size = len(child_json)

                print(f"DEBUG: Processing child: {child.get('name', 'Unknown')}")
                print(f"DEBUG: Child JSON size: {child_size} characters")
                print(f"DEBUG: Current chunk size: {current_chunk_size}")
                print(
                    f"DEBUG: Max capacity: {self._estimate_qr_code_capacity(max_version)}"
                )

                # Check if adding this child would exceed capacity
                if current_chunk_size + child_size > self._estimate_qr_code_capacity(
                    max_version
                ):
                    # Put the child back and stop adding
                    print(
                        f"DEBUG: Child {child.get('name', 'Unknown')} would exceed capacity. Stopping chunk."
                    )
                    remaining_children.insert(0, child)
                    break

                # Add the child to the chunk
                current_chunk["children"].append(child)
                current_chunk_size += child_size

            # Add chunk info
            current_chunk["chunk_info"] = {
                "chunk_number": len(chunks) + 1,
                "total_chunks": None,  # Will be updated later
            }

            chunks.append(current_chunk)
            print(
                f"DEBUG: Created chunk {len(chunks)} with {len(current_chunk['children'])} children"
            )

        # Update total chunks in each chunk
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["chunk_info"]["total_chunks"] = total_chunks

        print(f"DEBUG: Total chunks created: {total_chunks}")
        print(f"DEBUG: Remaining children: {len(remaining_children)}")

        return chunks

    def _traverse_folder(
        self, folder_path: str, base_path: str = None, max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Recursively traverse a folder and create a structured representation.
        """
        print(f"DEBUG: Traversing folder: {folder_path}")
        print(f"DEBUG: Current max depth: {max_depth}")

        # Initialize base path if not provided
        base_path = base_path or folder_path

        # Prevent excessive recursion
        if max_depth <= 0:
            print("DEBUG: Max depth reached. Returning empty structure.")
            return {}

        folder_structure = {
            "name": os.path.basename(folder_path),
            "path": os.path.relpath(folder_path, base_path),
            "type": "directory",
            "children": [],
            "metadata": {"total_files": 0, "total_size": 0},
        }

        try:
            # List all entries in the directory
            entries = list(os.scandir(folder_path))
            print(f"DEBUG: Found {len(entries)} entries in {folder_path}")

            for entry in entries:
                try:
                    # Handle files
                    if entry.is_file():
                        print(f"DEBUG: Processing file: {entry.path}")
                        file_info = {
                            "name": entry.name,
                            "path": os.path.relpath(entry.path, base_path),
                            "type": "file",
                            "size": entry.stat().st_size,
                        }

                        # Read file content
                        try:
                            with open(entry.path, "r", encoding="utf-8") as f:
                                file_info["content"] = f.read()
                            file_info["is_text"] = True
                            print(f"DEBUG: Successfully read text file: {entry.name}")
                        except UnicodeDecodeError:
                            # For binary files, base64 encode
                            print(f"DEBUG: Binary file detected: {entry.name}")
                            with open(entry.path, "rb") as f:
                                file_info["content"] = base64.b64encode(
                                    f.read()
                                ).decode("utf-8")
                            file_info["is_text"] = False

                        # Calculate file hash
                        file_info["hash"] = hashlib.md5(
                            file_info["content"].encode("utf-8")
                        ).hexdigest()

                        folder_structure["children"].append(file_info)
                        folder_structure["metadata"]["total_files"] += 1
                        folder_structure["metadata"]["total_size"] += file_info["size"]

                    # Recursively handle subdirectories
                    elif entry.is_dir():
                        print(f"DEBUG: Processing subdirectory: {entry.path}")
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
                    print(f"ERROR processing {entry.path}: {entry_error}")

        except Exception as folder_error:
            print(f"ERROR traversing folder {folder_path}: {folder_error}")
            return {}

        print(f"DEBUG: Folder traversal complete for {folder_path}")
        print(f"DEBUG: Total files: {folder_structure['metadata']['total_files']}")
        print(f"DEBUG: Total size: {folder_structure['metadata']['total_size']} bytes")

        return folder_structure

    def generate_folder_qr_code(
        self,
        folder_path: str,
        compress: bool = True,
        encrypt: bool = True,
        max_depth: int = 3,
        max_version: int = 40,
    ) -> Tuple[List[Any], List[str]]:
        """
        Generate multiple QR codes for a folder structure.
        """
        print(f"DEBUG: Starting folder QR code generation")
        print(f"DEBUG: Input folder: {folder_path}")
        print(f"DEBUG: Compression: {compress}")
        print(f"DEBUG: Encryption: {encrypt}")
        print(f"DEBUG: Max depth: {max_depth}")
        print(f"DEBUG: Max QR version: {max_version}")

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

        print(f"DEBUG: Generated folder metadata:")
        for key, value in metadata.items():
            print(f"DEBUG:   {key}: {value}")

        # Smart chunking of folder structure
        folder_chunks = self._smart_chunk_folder_structure(
            folder_structure, max_version=max_version
        )

        # Generate QR codes for each chunk
        qr_images = []
        encoded_chunks = []

        print(f"DEBUG: Preparing to generate QR codes for {len(folder_chunks)} chunks")

        for i, chunk in enumerate(folder_chunks, 1):
            print(f"DEBUG: Processing chunk {i}")

            # Convert chunk to JSON
            chunk_json = json.dumps(chunk)
            print(f"DEBUG: Chunk {i} JSON size: {len(chunk_json)} characters")

            # Optionally compress the chunk
            if compress:
                chunk_json = base64.b64encode(
                    zlib.compress(chunk_json.encode("utf-8"))
                ).decode("utf-8")
                print(f"DEBUG: Chunk {i} compressed size: {len(chunk_json)} characters")

            # Generate QR code for the chunk
            qr_image, encoded_chunk = self.generate_qr_code(
                chunk_json,
                compress=False,  # Already compressed if needed
                encrypt=encrypt,
                metadata=metadata,
            )

            qr_images.append(qr_image)
            encoded_chunks.append(encoded_chunk)

        print(f"DEBUG: QR code generation complete")
        print(f"DEBUG: Total QR codes generated: {len(qr_images)}")

        return qr_images, encoded_chunks


# Example usage demonstrating all processors
def main():
    # Text QR Processor Example
    text_processor = TextQRProcessor()
    original_text = "Hello, this is a test message for QR code generation!"

    text_qr_image, text_encoded_data = text_processor.generate_qr_code(
        original_text, compress=True, encrypt=True
    )
    text_processor.save_qr_code(text_qr_image, "text_qr_code.png")
    decoded_text = text_processor.decode_qr_text(text_encoded_data)
    print("Text Decoding Test:")
    print("Original Text:", original_text)
    print("Decoded Text:", decoded_text)
    print("Texts Match:", original_text == decoded_text)
    print()

    # File QR Processor Example
    file_processor = FileQRProcessor()
    test_file_path = "test_file.txt"

    # Create a test file
    with open(test_file_path, "w") as f:
        f.write("This is a test file content for QR code generation.")

    file_qr_image, file_encoded_data = file_processor.generate_file_qr_code(
        test_file_path, compress=False, encrypt=True
    )
    file_processor.save_qr_code(file_qr_image, "file_qr_code.png")

    # Reconstruct the file
    reconstructed_file_path = file_processor.reconstruct_file_from_qr(file_encoded_data)
    print("File Reconstruction Test:")
    print("Original File:", test_file_path)
    print("Reconstructed File:", reconstructed_file_path)

    # Compare file contents
    with open(test_file_path, "r") as f:
        original_content = f.read()

    with open(reconstructed_file_path, "r") as f:
        reconstructed_content = f.read()

    print("Files Match:", original_content == reconstructed_content)

    folder_processor = FolderQRProcessor()

    # Example folder to encode
    test_folder = "test"

    # Generate QR code for the folder
    folder_qr_image, folder_encoded_data = folder_processor.generate_folder_qr_code(
        test_folder, compress=True, encrypt=True, max_depth=2  # Limit depth to 2 levels
    )

    # Save QR code
    folder_processor.save_qr_code(folder_qr_image, "folder_qr_code.png")

    # Reconstruct folder
    reconstructed_folder = folder_processor.reconstruct_folder_from_qr(
        folder_encoded_data, output_dir=""
    )

    print(f"Folder reconstructed at: {reconstructed_folder}")


if __name__ == "__main__":
    main()
