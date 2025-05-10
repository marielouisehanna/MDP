import os
import ctypes

# 🔥 FORCE-LOAD the correct 64-bit zlibwapi.dll
dll_path = r"C:\Users\hcmon\Downloads\zlibwapi(1)\64bit\zlibwapi.dll"
ctypes.WinDLL(dll_path)

from spleeter.separator import Separator

def main():
    separator = Separator('spleeter:2stems')
    input_file = r"C:\Projects\ghanili\audio_files\law.wav"
    output_folder = r"C:\Projects\ghanili\outputs"
    separator.separate_to_file(input_file, output_folder)
    print("✅ Separation complete!")

if __name__ == "__main__":
    main()
