import streamlit as st
import requests
import io
import time
import dotenv
import os

dotenv.load_dotenv()

st.title("ExtractMe")
st.write("Upload your PDF and extract tables!")

# Subir archivo
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process File"):
        with st.spinner("Processing... Please wait.🕐"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(os.getenv("UPLOAD_ENDPOINT"), files=files, stream=True)

                if response.status_code == 200:
                    zip_buffer = io.BytesIO(response.content)

                    if zip_buffer.getbuffer().nbytes == 0:
                        st.error("ZIP file is empty. Check the backend for errors.")
                    else:
                        st.success("Processing complete! Download the extracted tables below.")

                        st.download_button(
                            label="Download ZIP",
                            data=zip_buffer,
                            file_name="extracted_data.zip",
                            mime="application/zip"
                        )
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")