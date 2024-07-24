import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import torch
from pyannote.audio import Pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment
import os
import threading

nltk.download('punkt', quiet=True)

class MeetingTranscriber:
    def __init__(self, master):
        self.master = master
        master.title("Meeting Transcriber")

        # File selection
        self.file_label = tk.Label(master, text="Select audio file:")
        self.file_label.pack()
        self.file_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.file_button.pack()

        # Transcribe button
        self.transcribe_button = tk.Button(master, text="Transcribe and Summarize", command=self.start_transcription)
        self.transcribe_button.pack()

        # Progress bar
        self.progress = tk.Label(master, text="")
        self.progress.pack()

        # Output text area
        self.output_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=20)
        self.output_text.pack()

        self.audio_path = None

    def browse_file(self):
        self.audio_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.audio_path:
            self.file_label.config(text=f"Selected file: {os.path.basename(self.audio_path)}")

    def start_transcription(self):
        if not self.audio_path:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        
        self.progress.config(text="Processing... This may take a while.")
        self.transcribe_button.config(state=tk.DISABLED)
        
        # Run transcription in a separate thread
        thread = threading.Thread(target=self.run_transcription)
        thread.start()

    def run_transcription(self):
        try:
            transcription = self.transcribe_audio_with_speaker_diarization(self.audio_path)
            full_text = " ".join([text for _, text in transcription])
            summary = self.summarize_text(full_text)
            key_points = self.extract_key_points(full_text)
            minutes = self.generate_minutes(transcription, summary, key_points)

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, minutes)

            with open("meeting_minutes.txt", "w") as f:
                f.write(minutes)

            self.progress.config(text="Transcription complete. Results saved to 'meeting_minutes.txt'")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.transcribe_button.config(state=tk.NORMAL)

    def transcribe_audio_with_speaker_diarization(self, audio_path):
        # (The rest of the function remains the same as in the previous version)
        # ...

    def summarize_text(self, text):
        # (The rest of the function remains the same as in the previous version)
        # ...

    def extract_key_points(self, text):
        # (The rest of the function remains the same as in the previous version)
        # ...

    def generate_minutes(self, transcription, summary, key_points):
        # (The rest of the function remains the same as in the previous version)
        # ...

if __name__ == "__main__":
    root = tk.Tk()
    app = MeetingTranscriber(root)
    root.mainloop()
