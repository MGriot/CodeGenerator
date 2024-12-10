# QR Code Master

## Description

A Python library for generating, processing, and reconstructing various graphical codes, including QR codes, DataMatrix, and barcodes, with features like compression, encryption, and error handling.

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Text QR Code

```python
from main2 import TextQRProcessor

# Create a TextQRProcessor instance
processor = TextQRProcessor()

# Generate a QR code from text
text = "Hello, QR Code!"
qr_image, encoded_data = processor.generate_qr_code(text, compress=True, encrypt=True)

# Save the QR code
processor.save_qr_code(qr_image, "text_qr.png")

# Decode the QR code
decoded_text = processor.decode_qr_text(encoded_data)
print(f"Decoded text: {decoded_text}")
```

### Using a Custom Encryption Key

```python
from main2 import TextQRProcessor
from cryptography.fernet import Fernet

# Generate a custom key (or load an existing one)
custom_key = Fernet.generate_key()

# Create a TextQRProcessor instance with the custom key
processor = TextQRProcessor(encryption_key=custom_key)

# Generate a QR code with encryption using the custom key
text = "This text is encrypted with a custom key."
qr_image, encoded_data = processor.generate_qr_code(text, encrypt=True)

# Save the QR code
processor.save_qr_code(qr_image, "custom_key_qr.png")

# To decode, you need to use the same custom key
processor_for_decoding = TextQRProcessor(encryption_key=custom_key)
decoded_text = processor_for_decoding.decode_qr_text(encoded_data)
print(f"Decoded text: {decoded_text}")
```

### File QR Code with Different File Types

```python
from main2 import FileQRProcessor

# Create a FileQRProcessor instance
processor = FileQRProcessor()

# Generate a QR code from a text file
txt_file_path = "my_text_file.txt"  # Replace with your text file path
qr_image_txt, encoded_data_txt = processor.generate_file_qr_code(
    txt_file_path, compress=False, encrypt=True
)
processor.save_qr_code(qr_image_txt, "text_file_qr.png")

# Generate a QR code from a binary file (e.g., an image)
image_file_path = "my_image.png"  # Replace with your image file path
qr_image_img, encoded_data_img = processor.generate_file_qr_code(
    image_file_path, compress=True, encrypt=False
)
processor.save_qr_code(qr_image_img, "image_file_qr.png")

# Reconstruct the files
reconstructed_txt_file = processor.reconstruct_file_from_qr(
    encoded_data_txt, output_dir="."
)
reconstructed_img_file = processor.reconstruct_file_from_qr(
    encoded_data_img, output_dir="."
)
print(f"Reconstructed text file: {reconstructed_txt_file}")
print(f"Reconstructed image file: {reconstructed_img_file}")
```

### Folder QR Code with Multiple Chunks

```python
from main2 import FolderQRProcessor

# Create a FolderQRProcessor instance
processor = FolderQRProcessor()

# Generate QR codes for a large folder (split into chunks)
large_folder_path = "my_large_folder"  # Replace with your large folder path
qr_images, encoded_data_chunks = processor.generate_folder_qr_code(
    large_folder_path, compress=True, encrypt=True, max_depth=3, max_chunk_size=5000
)

# Save the QR code chunks
for i, qr_image in enumerate(qr_images):
    processor.save_qr_code(qr_image, f"large_folder_qr_chunk_{i+1}.png")

# Reconstruct the folder from the QR code chunks
reconstructed_folder_path = processor.reconstruct_folder_from_qr_chunks(
    encoded_data_chunks, output_dir="."
)
print(f"Reconstructed folder saved to: {reconstructed_folder_path}")
```

## Error Handling

The library includes error handling for various scenarios:

*   **File Not Found:** When generating a QR code from a file or folder, if the specified path does not exist, a `FileNotFoundError` is raised.
*   **File Integrity Check:** When reconstructing a file, the library verifies its integrity using an MD5 hash. If the check fails, a `ValueError` is raised.
*   **Incorrect QR Data:** If the provided QR data is invalid or corrupted, the decoding process may raise exceptions like `json.JSONDecodeError` or `UnicodeDecodeError`.
*   **Encryption Key Mismatch:** If you attempt to decrypt data with an incorrect encryption key, a `cryptography.fernet.InvalidToken` exception is raised.

## Class Overview

*   **`QRCodeMasterProcessor`:** Base class for QR code processing, handling compression, encryption, and basic QR code generation/decoding.
*   **`TextQRProcessor`:** Inherits from `QRCodeMasterProcessor`. Specifically designed for encoding and decoding text data in QR codes.
*   **`FileQRProcessor`:** Inherits from `QRCodeMasterProcessor`. Handles QR code generation from files and reconstruction of files from QR codes.
*   **`FolderQRProcessor`:** Inherits from `FileQRProcessor`. Manages QR code generation for entire folder structures and their reconstruction, including handling large folders by splitting them into chunks.

## Encryption Key

The `QRCodeMasterProcessor` uses an encryption key for encrypting and decrypting data. If no key is provided during initialization, a new key is generated.

**Security Note:** It's crucial to manage the encryption key securely, especially when dealing with sensitive data. If you're using a custom key, ensure it's stored safely and backed up appropriately. Loss of the encryption key will result in the inability to decrypt the data.

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your branch to your forked repository.
5. Submit a pull request to the main repository.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
