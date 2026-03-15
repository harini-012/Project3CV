import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import random

st.set_page_config(layout="wide")
st.title("Interactive Background Removal App")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((512,512))
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Sidebar controls
    st.sidebar.header("Background Options & Controls")
    bg_option = st.sidebar.selectbox(
        "Choose Background Effect",
        ["--Select--", "Blur Original", "Random", "Black & White", "Sepia / Antique", "Solid Color", "Original"]
    )

    # Sliders
    blur_intensity = st.sidebar.slider("Blur Intensity", 0, 50, 25)
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)

    # Solid color picker appears only if Solid Color is selected
    if bg_option == "Solid Color":
        solid_color = st.sidebar.color_picker("Pick a solid color for background", "#00ff00")

    # Button to generate new random background
    new_bg = st.sidebar.button("Generate New Background")

    if bg_option != "--Select--":
        with st.spinner("Processing image..."):
            # Remove background
            output = remove(image)
            output_np = np.array(output)
            mask = output_np[:,:,3]
            foreground = output_np[:,:,:3]

            h, w = mask.shape
            mask_inv = cv2.bitwise_not(mask)

            # Adjust brightness/contrast for foreground
            fg_adj = cv2.convertScaleAbs(foreground, alpha=contrast, beta=(brightness-1)*100)

            # ------------------- Background Selection -------------------
            if bg_option == "Blur Original":
                bg = cv2.GaussianBlur(image_np, (blur_intensity*2+1, blur_intensity*2+1), 0)

            elif bg_option == "Random":
                choice = random.choice(["color","gradient","noise"]) if new_bg else "color"
                if choice == "color":
                    bg = np.full((h,w,3),
                                 (random.randint(0,255),
                                  random.randint(0,255),
                                  random.randint(0,255)),
                                 dtype=np.uint8)
                elif choice == "gradient":
                    bg = np.zeros((h,w,3), dtype=np.uint8)
                    c1 = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    c2 = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    for i in range(h):
                        alpha = i/h
                        bg[i,:] = (np.array(c1)*(1-alpha) + np.array(c2)*alpha).astype(np.uint8)
                else:
                    bg = np.random.randint(0,255,(h,w,3),dtype=np.uint8)

            elif bg_option == "Black & White":
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                _, bg = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

            elif bg_option == "Sepia / Antique":
                bg = image_np.copy()
                kernel = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
                bg = cv2.transform(bg, kernel)
                bg = np.clip(bg,0,255).astype(np.uint8)

            elif bg_option == "Solid Color":
                # Use the user-selected solid color
                c = tuple(int(solid_color.lstrip("#")[i:i+2],16) for i in (0,2,4))
                bg = np.full((h,w,3), c, dtype=np.uint8)

            else:
                bg = image_np

            # Combine foreground and background
            background = cv2.bitwise_and(bg, bg, mask=mask_inv)
            foreground = cv2.bitwise_and(fg_adj, fg_adj, mask=mask)
            final = cv2.add(background, foreground)
            final_image = Image.fromarray(final)

            st.subheader("Processed Image")
            st.image(final_image, use_container_width=True)

            # Download button
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )