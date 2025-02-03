import streamlit as st
import audio  # Import the audio module

# Set page configuration as the first Streamlit command
st.set_page_config(layout='wide', page_title="Audio Detection App", page_icon="🎵")

def main():
    audio.main()  # Directly call the main function from audio.py

if __name__ == '__main__':
    main()
