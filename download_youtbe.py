import yt_dlp
import ffmpeg
import os

def download_youtube_audio(url):
    save_path = os.path.dirname(os.path.abspath(__file__))  # Current directory path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Ensure the directory exists

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Changed to WAV audio
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),  # Save the file in the specified directory
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return os.path.join(save_path, ydl.prepare_filename(info_dict).replace('.webm', '.wav'))  # Ensure it's WAV

##def convert_m4a_to_mp3(input_file, output_file):
##    ffmpeg.input(input_file).output(output_file, acodec='libmp3lame', ab='192k').run()

if __name__ == "__main__":
    # Read URLs from file
    with open("urls.txt", "r") as file:
        urls = [line.strip() for line in file if line.strip()]
    
    for url in urls:
        filepath = download_youtube_audio(url)
        print(f"Downloaded audio file: {filepath}")

    # The conversion part is commented out in your original code
    # but if you want to use it, you would need to modify it for WAV files