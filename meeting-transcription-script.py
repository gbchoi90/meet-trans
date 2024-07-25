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

            with open("meeting_minutes.txt", "w", encoding='utf-8') as f:
                f.write(minutes)

            self.progress.config(text="Transcription complete. Results saved to 'meeting_minutes.txt'")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.transcribe_button.config(state=tk.NORMAL)

    def transcribe_audio_with_speaker_diarization(self, audio_path):
        # Initialize the pyannote.audio pipeline
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

        # Apply the pipeline to the audio file
        diarization = diarization_pipeline(audio_path)

        # Load the audio file
        audio = AudioSegment.from_wav(audio_path)

        # Process the diarization results
        transcription = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = int(turn.start * 1000)
            end_time = int(turn.end * 1000)
            segment = audio[start_time:end_time]
            
            text = self.transcribe_audio_segment(segment)
            if text:
                transcription.append((speaker, text))

        return transcription

    def transcribe_audio_segment(self, audio_segment):
        recognizer = sr.Recognizer()
        audio_segment.export("temp.wav", format="wav")
        
        with sr.AudioFile("temp.wav") as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                text = ""
            except sr.RequestError:
                text = "[Error: Could not request results from speech recognition service]"
        
        os.remove("temp.wav")
        return text

    def summarize_text(self, text):
        # Load a model fine-tuned for meeting summarization
        model_name = "samvaity/meeting-summary-best-model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Prepare the text
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def extract_key_points(self, text):
        # Initialize KeyBERT for keyword extraction
        kw_model = KeyBERT()

        # Extract keywords
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.7)

        # Get the top 5 keywords
        top_keywords = [kw for kw, _ in keywords[:5]]

        # Find sentences containing these keywords
        sentences = sent_tokenize(text)
        key_points = []
        for keyword in top_keywords:
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    key_points.append(sentence)
                    break

        return key_points

    def generate_minutes(self, transcription, summary, key_points):
        # Get the current date and time
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M")
        
        # Start building the minutes
        minutes = f"Meeting Minutes - {date_string}\n\n"
        minutes += "Attendees: [List of attendees]\n\n"
        minutes += f"Summary:\n{summary}\n\n"
        minutes += "Key Points:\n"
        for point in key_points:
            minutes += f"- {point}\n"
        minutes += "\nDetailed Discussion:\n"
        
        for speaker, text in transcription:
            minutes += f"{speaker}: {text}\n"
        
        minutes += "\nAction Items:\n[List action items here]\n"
        minutes += "\nNext Meeting: [Date and Time of Next Meeting]"
        
        return minutes

if __name__ == "__main__":
    root = tk.Tk()
    app = MeetingTranscriber(root)
    root.mainloop()