import qrcode
from PIL import Image


def create_vcard(name, surname, position, email, factory, phone_number, address):
    """
    Create a vCard string with the given contact information
    """
    vcard = f"BEGIN:VCARD\n"
    vcard += f"VERSION:3.0\n"
    vcard += f"FN:{name} {surname}\n"
    vcard += f"LN:{surname};FN:{name}\n"
    vcard += f"TITLE:{position}\n"
    vcard += f"EMAIL;TYPE=work,INTERNET:{email}\n"
    vcard += f"TEL;TYPE=work,VOICE:{phone_number}\n"
    vcard += f"ORG:{factory}\n"
    vcard += f"ADR;TYPE=work:;;{address}\n"
    vcard += f"END:VCARD"
    return vcard


def generate_qr_code(vcard, logo_path):
    """
    Generate a QR code from the vCard string with a logo in the middle
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(vcard)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Open the logo image
    logo = Image.open(logo_path)

    # Calculate the position to place the logo
    pos = ((img.size[0] - logo.size[0]) // 2, (img.size[1] - logo.size[1]) // 2)

    # Paste the logo onto the QR code
    img.paste(logo, pos)

    return img


# Example usage:
name = "John"
surname = "Doe"
position = "Software Engineer"
email = "john.doe@example.com"
factory = "Example Factory"
phone_number = "+1234567890"
address = "123 Main St, Anytown, USA 12345"
logo_path = r"C:\Users\Admin\Documents\Coding\CodeGenerator\contact_qr_code.png"

vcard = create_vcard(name, surname, position, email, factory, phone_number, address)
qr_code = generate_qr_code(vcard, logo_path)

# Save the QR code to a file
qr_code.save("contact_qr_code_with_logo.png")
