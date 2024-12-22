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

    def _encode_base64_str(self, data: bytes) -> str:
        """Convert bytes to base64 string"""
        return base64.b64encode(data).decode('utf-8')

    def generate_qr_code(
        self,
        data: Union[str, bytes],
        compress: bool = False,
        encrypt: bool = False,
        metadata: Dict[str, Any] = None,
        max_chunk_size: int = 1000
    ) -> Tuple[List[Image.Image], List[str]]:
        if metadata is None:
            metadata = {}
        metadata["compressed"] = compress
        metadata["encrypted"] = encrypt

        # Convert input to bytes
        processed_data = data.encode("utf-8") if isinstance(data, str) else data

        # Process data based on flags
        if compress:
            processed_data = self.compress_data(processed_data)
            processed_data = self._encode_base64_str(processed_data)

        if encrypt:
            processed_data = self.encrypt_data(processed_data if isinstance(processed_data, bytes) else processed_data.encode('utf-8'))
            processed_data = self._encode_base64_str(processed_data)

        if not (compress or encrypt):
            processed_data = processed_data.decode('utf-8') if isinstance(processed_data, bytes) else processed_data

        # Create QR data
        qr_data = json.dumps({"metadata": metadata, "content": processed_data})
        chunks = self._split_data_for_qr(qr_data, max_chunk_size)

        qr_images = []
        encoded_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_number"] = i + 1
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_data = json.dumps({"metadata": chunk_metadata, "content": chunk})

            version = self._calculate_qr_version(chunk_data)
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
    pass


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
def main():
    # Text QR Processor Example
    text_processor = TextQRProcessor()
    original_text = "Hello, this is a test message for QR code generation!"

    text_qr_image, text_encoded_data = text_processor.generate_qr_code(
        original_text, compress=True, encrypt=True
    )
    text_processor.save_qr_code(text_qr_image, "text_qr_code.png")
    decoded_text = text_processor.decode_qr_data(text_encoded_data)
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
    print()

    # Folder QR Processor Example
    folder_processor = FolderQRProcessor()
    test_folder = "test"

    # Generate QR codes for the folder
    folder_qr_images, folder_encoded_chunks = folder_processor.generate_folder_qr_code(
        test_folder, compress=True, encrypt=True, max_depth=2  # Limit depth to 2 levels
    )

    # Save QR codes
    for i, qr_image in enumerate(folder_qr_images):
        folder_processor.save_qr_code(qr_image, f"folder_qr_code_{i}.png")

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
        folder_processor.save_qr_code(qr_image, f"folder_qr_code_chunk_{i}.png")

    # Reconstruct folder from chunks
    reconstructed_folder_chunk = folder_processor.reconstruct_folder_from_qr_chunks(
        encoded_chunks, output_dir="."
    )

    print(f"Chunked folder reconstructed at: {reconstructed_folder_chunk}")


if __name__ == "__main__":
    main()
