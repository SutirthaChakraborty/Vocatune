#!/usr/bin/env python3
import asyncio
import threading
import threading
import numpy as np
import pyaudio
import mido
import tkinter as tk
from tkinter import ttk, messagebox
from mido import Message
import time
import queue
from PIL import Image, ImageTk

from ultralytics import YOLO
import cv2
import numpy as np
import time
import random
import math
import mediapipe as mp
import cv2
import simpleaudio as sa  # For metronome sound

# Set up MIDI output
# Ensure that a virtual MIDI port named 'PythonMIDI' is created in your MIDI settings
try:
    midi_out = mido.open_output('PythonMIDI', virtual=True)
except IOError:
    print(
        "Error: Could not open MIDI port 'PythonMIDI'. Make sure it is created in your MIDI settings."
    )
    exit(1)

# Define musical scales
SCALES = {
    'C Major': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'A Minor': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'G Major': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    'E Minor': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
    'D Major': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    'B Minor': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
    'A Major': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
    'F# Minor': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
    'E Major': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
    'C# Minor': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
    'F Major': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
    'D Minor': ['D', 'E', 'F', 'G', 'A', 'Bb', 'C'],
    'Bb Major': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
    'G Minor': ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
    'Eb Major': ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
    'C Minor': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
    'Ab Major': ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
    'F Minor': ['F', 'G', 'Ab', 'Bb', 'C', 'Db', 'Eb'],
    'Db Major': ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
    'Bb Minor': ['Bb', 'C', 'Db', 'Eb', 'F', 'Gb', 'Ab'],
}


# Convert note names to MIDI note numbers
NOTE_TO_MIDI = {
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
}

# Chord definitions
CHORDS = {
    'Major': [0, 4, 7],
    'Minor': [0, 3, 7],
    'Major 7th': [0, 4, 7, 11],
    'Minor 7th': [0, 3, 7, 10],
}

# List of possible root notes for chords
ROOT_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class YOLOv8PoseModel:
    def __init__(self, model_path='yolo11m-pose.pt'):
        self.model = YOLO(model_path)  # Load YOLOv8-pose model

    def detect_pose(self, frame):
        # Run pose detection on the frame
        results = self.model.predict(
            frame, conf=0.3, verbose=False
        )  # confidence threshold = 0.3
        if results and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints  # Access keypoints from the results
            return keypoints
        return None


# Helper functions
def closest_note_in_scale(pitch_midi, scale_notes_midi):
    differences = [abs(pitch_midi - note) for note in scale_notes_midi]
    min_diff_index = differences.index(min(differences))
    return scale_notes_midi[min_diff_index]


def freq_to_midi(frequency):
    if frequency <= 0:
        return None
    return 69 + 12 * np.log2(frequency / 440.0)


def yin_pitch_detection(audio_buffer, sample_rate, threshold):
    buffer_size = len(audio_buffer)
    tau_max = buffer_size // 2

    # Difference function
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff[tau] = np.sum(
            (audio_buffer[: buffer_size - tau] - audio_buffer[tau:buffer_size]) ** 2
        )

    # Cumulative mean normalized difference function
    cmndf = np.zeros(tau_max)
    cmndf[0] = 1
    running_sum = 0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmndf[tau] = diff[tau] * tau / running_sum if running_sum != 0 else 1

    # Absolute threshold
    tau = 1
    while tau < tau_max:
        if cmndf[tau] < threshold:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            pitch = sample_rate / tau
            return pitch
        tau += 1

    return None


class AutoTuneApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Autotune MIDI Controller")
        self.gui_queue = queue.Queue()
        self.yolo_model = YOLOv8PoseModel()

        # Basic variables
        self.metronome_wave = self.generate_metronome_click(frequency=1000)
        self.running = True
        self.exit_event = threading.Event()
        self.current_chord_var = tk.StringVar(value="")
        try:
            self.loop = asyncio.new_event_loop()
        except:
            self.loop = asyncio.get_event_loop()

        self.thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self.thread.start()

        self.scale_var = tk.StringVar(value='C Major')
        self.tempo_var = tk.IntVar(value=120)
        self.audio_silence_threshold_var = tk.DoubleVar(value=0.01)
        self.yin_threshold_var = tk.DoubleVar(value=0.15)
        self.metronome_volume_var = tk.DoubleVar(value=0.011)
        self.octave_var = tk.IntVar(value=4)

        # Create a custom style for colored buttons
        self.style = ttk.Style()
        self.style.theme_use('default')  # Use the default theme as a base
        self.style.configure("Green.TButton", background="green", foreground="white")
        self.style.configure("Red.TButton", background="red", foreground="white")
        self.style.map(
            "Green.TButton",
            background=[('active', 'dark green'), ('pressed', 'green')],
            foreground=[('active', 'white'), ('pressed', 'white')],
        )
        self.style.map(
            "Red.TButton",
            background=[('active', 'dark red'), ('pressed', 'red')],
            foreground=[('active', 'white'), ('pressed', 'white')],
        )

        # Add these lines to the __init__ method
        self.left_hand_positions = []
        self.right_hand_positions = []
        self.max_positions = 5  # Number of positions to keep for the moving average

        # Ensure all channel toggle variables are initialized
        for i in range(1, 8):
            setattr(self, f'channel{i}_enabled', tk.BooleanVar(value=True))
        # Separate octave variables for each channel
        self.octave_var_ch1 = tk.IntVar(value=4)
        self.octave_var_ch2 = tk.IntVar(value=4)
        self.octave_var_ch3 = tk.IntVar(value=4)
        self.octave_var_ch4 = tk.IntVar(value=4)  # Added for Channel 4
        self.octave_var_ch5 = tk.IntVar(value=4)  # Added for Channel 5
        # Add variables for arm positions
        self.right_arm_x = tk.DoubleVar(value=0.5)
        self.right_arm_y = tk.DoubleVar(value=0.5)
        self.left_arm_x = tk.DoubleVar(value=0.5)
        self.left_arm_y = tk.DoubleVar(value=0.5)

        # GUI variables
        self.channel1_detected_pitch = tk.DoubleVar(value=0.0)
        self.channel1_corrected_pitch = tk.StringVar(value="None")
        self.channel2_velocity = tk.IntVar(value=0)
        self.channel3_velocity = tk.IntVar(value=0)
        self.current_bar = 1
        self.current_chord_label = tk.StringVar(value="")

        # Hand tracking variables
        self.right_hand_over_head_start = None
        self.left_hand_over_head_start = None

        # Channel enable/disable variables
        self.channel1_enabled = tk.BooleanVar(value=True)  # Add this line
        self.channel2_enabled = tk.BooleanVar(value=True)
        self.channel3_enabled = tk.BooleanVar(value=True)
        self.channel4_enabled = tk.BooleanVar(value=True)  # Added for Channel 4
        self.channel5_enabled = tk.BooleanVar(value=True)  # Added for Channel 5

        # Send full chords options
        self.send_chords_ch1_var = tk.BooleanVar(value=False)
        self.send_chords_ch2_var = tk.BooleanVar(value=False)
        self.send_chords_ch3_var = tk.BooleanVar(value=False)
        self.send_chords_ch4_var = tk.BooleanVar(value=False)  # Added for Channel 4
        self.send_chords_ch5_var = tk.BooleanVar(value=False)  # Added for Channel 5

        # Hand control toggles
        self.right_hand_control_enabled = True
        self.left_hand_control_enabled = True

        self.frame_width = 0
        self.frame_height = 0

        # Chord selection variables
        self.chord1_var = tk.StringVar(value='C Major')
        self.chord2_var = tk.StringVar(value='F Major')
        self.chord3_var = tk.StringVar(value='G Major')
        self.chord4_var = tk.StringVar(value='C Major')

        # Other variables
        self.selected_scale = self.scale_var.get()
        self.root_midi_note = self.get_root_midi(self.selected_scale)
        self.last_note_channel1 = None
        self.last_notes_channel2 = []
        self.last_notes_channel3 = []
        self.last_notes_channel4 = []  # Added for Channel 4
        self.last_notes_channel5 = []  # Added for Channel 5
        self.current_chord_index = 0

        # Initialize note_on_sent flags and velocities for each channel
        self.note_on_sent_channel2 = False
        self.last_velocity_channel2 = 0
        self.note_on_sent_channel3 = False
        self.last_velocity_channel3 = 0
        self.note_on_sent_channel4 = False  # Added for Channel 4
        self.note_on_sent_channel5 = False  # Added for Channel 5

        # Add new variables for channels 4 and 5
        self.channel4_enabled = tk.BooleanVar(value=True)
        self.channel5_enabled = tk.BooleanVar(value=True)
        self.send_chords_ch4_var = tk.BooleanVar(value=False)
        self.send_chords_ch5_var = tk.BooleanVar(value=False)

        # Add new variables for guitar and piano channels
        self.channel6_enabled = tk.BooleanVar(value=True)
        self.channel7_enabled = tk.BooleanVar(value=True)
        self.octave_var_ch6 = tk.IntVar(value=4)
        self.octave_var_ch7 = tk.IntVar(value=5)
        self.guitar_pattern = []
        self.piano_pattern = []

        self.channel_info = {
            1: tk.StringVar(value="Channel 1 (Voice)"),
            2: tk.StringVar(value="Channel 2 (Right Hand)"),
            3: tk.StringVar(value="Channel 3 (Left Hand)"),
            4: tk.StringVar(value="Channel 4: Chord"),
            5: tk.StringVar(value="Channel 5: Chord"),
            6: tk.StringVar(value="Channel 6 (Guitar): Strum"),
            7: tk.StringVar(value="Channel 7 (Piano): Strum"),
        }
        # Build chord progression
        self.chord_progression = self.build_chord_progression()

        # Initialize current_chord_notes for each channel with their respective octaves
        self.current_chord_notes_channel1 = self.get_current_chord_notes(
            self.octave_var_ch1.get()
        )
        self.current_chord_notes_channel2 = self.get_current_chord_notes(
            self.octave_var_ch2.get()
        )
        self.current_chord_notes_channel3 = self.get_current_chord_notes(
            self.octave_var_ch3.get()
        )

        # Metronome
        self.metronome_interval = None
        self.metronome_playing = False
        self.metronome_wave = self.generate_metronome_click()

        # Setup GUI
        self.setup_gui()
        self.setup_callbacks()

        # Start threads
        self.audio_thread = threading.Thread(target=self.audio_process)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        self.video_thread = threading.Thread(target=self.video_process)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Separate calibration variables for each hand
        self.left_min_x, self.left_max_x = float('inf'), float('-inf')
        self.left_min_y, self.left_max_y = float('inf'), float('-inf')
        self.right_min_x, self.right_max_x = float('inf'), float('-inf')
        self.right_min_y, self.right_max_y = float('inf'), float('-inf')

        self.calibration_duration = 10  # 5 seconds for calibration
        self.is_calibrating = False
        self.calibration_start_time = None

        # Start metronome
        self.start_metronome()

        # Start GUI update loop
        self.update_gui()

    def generate_metronome_click(
        self, frequency=1000, duration=0.05, sample_rate=44100
    ):
        """
        Generates a simple click sound for the metronome.
        Args:
            frequency: The frequency of the sound in Hz.
            duration: The duration of the click in seconds.
            sample_rate: The sample rate in Hz.
        Returns:
            A numpy array representing the sound wave.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        click_wave = np.sin(frequency * t * 2 * np.pi)
        click_wave = (click_wave * 32767).astype(np.int16)
        return click_wave

    def setup_gui(self):
        self.root.title("Real-Time Autotune MIDI Controller")
        self.root.geometry(
            "1400x1000"
        )  # Adjusted window size to fit all elements including the webcam feed

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Global Settings Frame (for scale, octave, tempo)
        global_frame = ttk.LabelFrame(main_frame, text="Global Settings")
        global_frame.grid(column=0, row=0, sticky=(tk.W, tk.E), pady=10, padx=10)

        ttk.Label(global_frame, text="Scale:").grid(column=0, row=0, sticky=tk.W)
        scale_menu = ttk.Combobox(
            global_frame, textvariable=self.scale_var, values=list(SCALES.keys())
        )
        scale_menu.grid(column=1, row=0, sticky=(tk.W, tk.E))

        ttk.Label(global_frame, text="Global Octave:").grid(
            column=2, row=0, sticky=tk.W
        )
        ttk.Spinbox(
            global_frame, from_=1, to=7, textvariable=self.octave_var, width=5
        ).grid(column=3, row=0, sticky=tk.W)

        ttk.Label(global_frame, text="Tempo (BPM):").grid(column=0, row=1, sticky=tk.W)
        ttk.Scale(
            global_frame,
            from_=60,
            to=300,
            orient='horizontal',
            variable=self.tempo_var,
            command=self.update_tempo,
        ).grid(column=1, row=1, sticky=(tk.W, tk.E))
        ttk.Label(global_frame, textvariable=self.tempo_var).grid(
            column=2, row=1, sticky=tk.W
        )

        # Add the label for displaying the current beat
        info_frame = ttk.Frame(global_frame)
        info_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(info_frame, text="Current Beat:").grid(column=0, row=0, sticky=tk.W)
        self.current_beat_label = ttk.Label(info_frame, text="")
        self.current_beat_label.grid(column=1, row=0, sticky=tk.W)
        ttk.Label(info_frame, text="Current Chord:").grid(
            column=2, row=0, sticky=tk.W, padx=(20, 0)
        )
        ttk.Label(info_frame, textvariable=self.current_chord_var).grid(
            column=3, row=0, sticky=tk.W
        )

        # Chord Progression Frame
        chord_frame = ttk.LabelFrame(main_frame, text="Chord Progression")
        chord_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), pady=10, padx=10)

        chord_options = self.get_chord_options()
        self.chord_menus = []
        for i, chord_var in enumerate(
            [self.chord1_var, self.chord2_var, self.chord3_var, self.chord4_var]
        ):
            ttk.Label(chord_frame, text=f"Chord {i+1}:").grid(
                column=i, row=0, sticky=tk.W
            )
            chord_menu = ttk.Combobox(
                chord_frame, textvariable=chord_var, values=chord_options
            )
            chord_menu.grid(column=i, row=1, sticky=(tk.W, tk.E), padx=5)
            self.chord_menus.append(chord_menu)

        # Channel Controls Frame
        channels_frame = ttk.LabelFrame(main_frame, text="Channel Controls")
        channels_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), pady=10, padx=10)

        for i in range(1, 8):  # Channels 1-7
            channel_frame = ttk.Frame(channels_frame)
            channel_frame.grid(column=0, row=i - 1, sticky=(tk.W, tk.E), pady=5)

            ttk.Label(channel_frame, text=f"Channel {i}:").grid(
                column=0, row=0, sticky=tk.W
            )

            if i == 1:
                ttk.Label(channel_frame, text="Voice Input").grid(
                    column=1, row=0, sticky=tk.W
                )
            elif i == 2:
                ttk.Label(channel_frame, text="Right Hand").grid(
                    column=1, row=0, sticky=tk.W
                )
            elif i == 3:
                ttk.Label(channel_frame, text="Left Hand").grid(
                    column=1, row=0, sticky=tk.W
                )
            elif i == 6:
                ttk.Label(channel_frame, text="Guitar").grid(
                    column=1, row=0, sticky=tk.W
                )
            elif i == 7:
                ttk.Label(channel_frame, text="Piano").grid(
                    column=1, row=0, sticky=tk.W
                )
            else:
                ttk.Label(channel_frame, text="Synth").grid(
                    column=1, row=0, sticky=tk.W
                )

            ttk.Label(channel_frame, text="Octave:").grid(column=2, row=0, sticky=tk.W)
            ttk.Spinbox(
                channel_frame,
                from_=1,
                to=7,
                textvariable=getattr(self, f'octave_var_ch{i}'),
                width=5,
            ).grid(column=3, row=0, sticky=tk.W)

            channel_button = ttk.Button(
                channel_frame,
                text="Disable",
                command=getattr(self, f'toggle_channel{i}'),
                width=10,
                style="Green.TButton",
            )
            channel_button.grid(column=4, row=0, sticky=tk.W)
            setattr(self, f'channel{i}_button', channel_button)

            if i not in [6, 7]:  # Not for Guitar and Piano
                ttk.Checkbutton(
                    channel_frame,
                    text="Send Full Chords",
                    variable=getattr(self, f'send_chords_ch{i}_var'),
                ).grid(column=5, row=0, sticky=tk.W)

            if i in [2, 3]:  # Only for hand-controlled channels
                ttk.Label(channel_frame, text="Velocity:").grid(
                    column=6, row=0, sticky=tk.W
                )
                ttk.Progressbar(
                    channel_frame,
                    orient='horizontal',
                    length=100,
                    mode='determinate',
                    variable=getattr(self, f'channel{i}_velocity'),
                    maximum=127,
                ).grid(column=7, row=0, sticky=(tk.W, tk.E))

        # Modify the Hand Control Visuals Frame
        hand_control_frame = ttk.LabelFrame(main_frame, text="Hand Controls")
        hand_control_frame.grid(column=0, row=3, sticky=(tk.W, tk.E), pady=10, padx=10)

        # Add Calibration Button
        self.calibrate_button = ttk.Button(
            hand_control_frame,
            text="Calibrate Hand Range",
            command=self.calibrate_hand_range,
        )
        self.calibrate_button.grid(column=0, row=0, columnspan=3, pady=(0, 10))

        # Left Hand (Channel 3) Graph
        left_hand_frame = ttk.LabelFrame(
            hand_control_frame, text="Left Hand Control (Channel 3)"
        )
        left_hand_frame.grid(
            column=0, row=1, padx=(0, 5), sticky=(tk.N, tk.W, tk.E, tk.S)
        )

        self.left_hand_canvas = tk.Canvas(
            left_hand_frame, width=300, height=300, bg='white'
        )
        self.left_hand_canvas.grid(column=0, row=0)
        self.left_hand_dot = self.left_hand_canvas.create_oval(
            0, 0, 10, 10, fill='blue'
        )
        self.setup_hand_canvas(self.left_hand_canvas, is_left=True)

        # Webcam Feed Frame (positioned between left and right canvases)
        webcam_frame = ttk.LabelFrame(hand_control_frame, text="Webcam Feed")
        webcam_frame.grid(column=1, row=1, padx=5, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Canvas for displaying webcam feed (same size as hand canvases)
        self.webcam_canvas = tk.Canvas(webcam_frame, width=300, height=300)
        self.webcam_canvas.grid(column=0, row=0, sticky=(tk.W, tk.E))

        # Right Hand (Channel 2) Graph
        right_hand_frame = ttk.LabelFrame(
            hand_control_frame, text="Right Hand Control (Channel 2)"
        )
        right_hand_frame.grid(
            column=2, row=1, padx=(5, 0), sticky=(tk.N, tk.W, tk.E, tk.S)
        )
        self.right_hand_canvas = tk.Canvas(
            right_hand_frame, width=300, height=300, bg='white'
        )
        self.right_hand_canvas.grid(column=0, row=0)
        self.right_hand_dot = self.right_hand_canvas.create_oval(
            0, 0, 10, 10, fill='red'
        )
        self.setup_hand_canvas(self.right_hand_canvas, is_left=False)

        # Advanced settings Frame (Metronome and pitch thresholds)
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Settings")
        advanced_frame.grid(column=0, row=4, sticky=(tk.W, tk.E), pady=10, padx=10)

        ttk.Label(advanced_frame, text="Metronome Volume:").grid(
            column=0, row=0, sticky=tk.W
        )
        ttk.Scale(
            advanced_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.metronome_volume_var,
        ).grid(column=1, row=0, sticky=(tk.W, tk.E))
        ttk.Label(advanced_frame, textvariable=self.metronome_volume_var).grid(
            column=2, row=0, sticky=tk.W
        )

        ttk.Label(advanced_frame, text="Audio Silence Threshold:").grid(
            column=0, row=1, sticky=tk.W
        )
        ttk.Scale(
            advanced_frame,
            from_=0.001,
            to=0.05,
            orient='horizontal',
            variable=self.audio_silence_threshold_var,
        ).grid(column=1, row=1, sticky=(tk.W, tk.E))
        ttk.Label(advanced_frame, textvariable=self.audio_silence_threshold_var).grid(
            column=2, row=1, sticky=tk.W
        )

        ttk.Label(advanced_frame, text="YIN Pitch Threshold:").grid(
            column=0, row=2, sticky=tk.W
        )
        ttk.Scale(
            advanced_frame,
            from_=0.05,
            to=0.3,
            orient='horizontal',
            variable=self.yin_threshold_var,
        ).grid(column=1, row=2, sticky=(tk.W, tk.E))
        ttk.Label(advanced_frame, textvariable=self.yin_threshold_var).grid(
            column=2, row=2, sticky=tk.W
        )

        # Add Exit button
        exit_button = ttk.Button(
            main_frame, text="Exit", command=self.on_closing, style="Red.TButton"
        )
        exit_button.grid(column=0, row=5, sticky=(tk.W, tk.E), pady=10)

        # Configure padding and weights
        for child in main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Make the window resizable
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def setup_hand_canvas(self, canvas, is_left):
        canvas.create_text(
            150, 15, text="Octave", anchor=tk.N, font=('TkDefaultFont', 10, 'bold')
        )
        canvas.create_text(
            15 if is_left else 285,
            150,
            text="Velocity",
            anchor=tk.W if is_left else tk.E,
            angle=90,
            font=('TkDefaultFont', 10, 'bold'),
        )

        for i in range(8):
            canvas.create_text(
                i * 37.5 + 37.5, 285, text=str(7 - i if is_left else i), anchor=tk.S
            )
            canvas.create_text(
                15 if is_left else 285,
                300 - i * 37.5 - 37.5,
                text=str(i * 16),
                anchor=tk.E if is_left else tk.W,
            )

        canvas.create_text(
            150,
            295,
            text="Left <- X-axis -> Right",
            anchor=tk.S,
            font=('TkDefaultFont', 8),
        )
        canvas.create_text(
            5 if is_left else 295,
            150,
            text="Low <- Y-axis -> High",
            anchor=tk.W if is_left else tk.E,
            angle=90,
            font=('TkDefaultFont', 8),
        )

    def generate_guitar_pattern(self):
        """Generate a random guitar strum pattern"""
        pattern = []
        for _ in range(8):  # 8 eighth notes in a bar
            if random.random() < 0.7:  # 70% chance of a strum
                pattern.append(random.choice(['down', 'up']))
            else:
                pattern.append('rest')
        return pattern

    def generate_piano_pattern(self):
        """Generate a random piano chord pattern"""
        pattern = []
        for _ in range(4):  # 4 quarter notes in a bar
            if random.random() < 0.8:  # 80% chance of a chord
                pattern.append(random.choice(['block', 'arpeggio']))
            else:
                pattern.append('rest')
        return pattern

    def stop_midi_channel(self, channel_index):
        for note in range(128):  # MIDI notes range from 0 to 127
            midi_out.send(
                Message('note_off', note=note, velocity=0, channel=channel_index)
            )

    def setup_callbacks(self):
        # Update chord progression when scale or chord selection changes
        self.scale_var.trace('w', self.on_scale_change)
        self.chord1_var.trace('w', self.on_chord_change)
        self.chord2_var.trace('w', self.on_chord_change)
        self.chord3_var.trace('w', self.on_chord_change)
        self.chord4_var.trace('w', self.on_chord_change)

    def get_chord_options(self):
        chords = []
        for root in ROOT_NOTES:
            for chord_type in CHORDS.keys():
                chords.append(f"{root} {chord_type}")
        return chords

    def parse_chord(self, chord_str, octave):
        """
        Parses a chord string like 'C Major' or 'C Major 7th' and returns the MIDI notes for the chord.
        """
        try:
            root_str, chord_type = chord_str.split(' ', 1)
        except ValueError:
            return []

        if root_str not in NOTE_TO_MIDI or chord_type not in CHORDS:
            return []

        root_note = NOTE_TO_MIDI[root_str]
        intervals = CHORDS[chord_type]
        chord_notes = [
            root_note + interval + (octave + 1) * 12 for interval in intervals
        ]  # MIDI note numbers
        return chord_notes

    def update_hand_position(self, canvas, dot, x, y):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        dot_size = 10

        # Determine if it's the left or right hand based on the canvas
        is_left_hand = canvas == self.left_hand_canvas

        # For left hand, invert the x-axis
        # For right hand, invert the x-axis as well
        x_pos = (1 - x) * (canvas_width - dot_size)

        # Y-axis remains the same for both hands
        y_pos = y * (canvas_height - dot_size)

        canvas.coords(dot, x_pos, y_pos, x_pos + dot_size, y_pos + dot_size)

    def on_scale_change(self, *args):
        self.selected_scale = self.scale_var.get()
        self.root_midi_note = self.get_root_midi(self.selected_scale)
        chord_options = self.get_chord_options()
        # Update chord option menus
        for i, menu in enumerate(self.chord_menus):
            self.update_option_menu(
                menu, chord_options, getattr(self, f'chord{i+1}_var')
            )
        # Update chord progression
        self.on_chord_change()

    def update_option_menu(self, option_menu, options, variable):
        option_menu['values'] = options
        if variable.get() not in options:
            variable.set(options[0])

    def on_chord_change(self, *args):
        self.chord_progression = [
            self.chord1_var.get(),
            self.chord2_var.get(),
            self.chord3_var.get(),
            self.chord4_var.get(),
        ]
        self.current_chord_index = 0
        self.current_chord_notes_channel2 = self.get_current_chord_notes(
            self.octave_var_ch2.get()
        )
        self.current_chord_notes_channel3 = self.get_current_chord_notes(
            self.octave_var_ch3.get()
        )

    def build_chord_progression(self):
        """
        Builds a chord progression list based on the selected chords.
        """
        progression = [
            self.chord1_var.get(),
            self.chord2_var.get(),
            self.chord3_var.get(),
            self.chord4_var.get(),
        ]
        return progression

    def get_current_chord_notes(self, octave):
        """
        Returns the MIDI note numbers for the current chord in the specified octave.
        """
        chord_str = self.chord_progression[self.current_chord_index]
        chord_notes = self.parse_chord(chord_str, octave)
        return chord_notes

    def get_root_midi(self, scale_name):
        """
        Get the MIDI note number for the root of the selected scale.
        """
        root_note_name = SCALES[scale_name][0]
        midi_number = NOTE_TO_MIDI.get(root_note_name, 0)  # Default to C if not found
        octave = self.octave_var.get()
        midi_number += (octave + 1) * 12  # Adjust for octave
        return midi_number

    def get_note_name(self, midi_number):
        """
        Returns the note name for a given MIDI number.
        """
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_index = midi_number % 12
        octave = (midi_number // 12) - 1
        note_name = note_names[note_index]
        return f"{note_name}{octave}"

    def start_metronome(self):
        """
        Starts the metronome based on the current tempo.
        """
        if self.metronome_playing:
            return
        self.metronome_playing = True
        tempo = int(self.tempo_var.get())
        self.metronome_interval = 60 / tempo
        threading.Thread(target=self.play_metronome).start()
        self.schedule_chord_change()

    def play_metronome(self):
        """
        Plays the metronome sounds on each beat. The first beat in the bar
        has a different sound than the other beats.
        """
        beat_count = 0
        # Generate two different sounds: one for the downbeat and one for the rest
        downbeat_wave = self.generate_metronome_click(
            frequency=1500
        )  # Higher-pitched "tik"
        beat_wave = self.generate_metronome_click(frequency=200)  # Lower-pitched "tak"
        while self.metronome_playing:
            volume = self.metronome_volume_var.get()
            if beat_count == 0:  # First beat (downbeat)
                metronome_wave = (downbeat_wave * volume).astype(np.int16)
            else:  # Other beats
                metronome_wave = (beat_wave * volume).astype(np.int16)

            # Play the metronome sound
            sa.play_buffer(
                metronome_wave.tobytes(),
                1,  # num_channels
                2,  # bytes_per_sample
                44100,  # sample_rate
            )

            beat_count = (
                beat_count % 4
            ) + 1  # Reset after 4 beats (a 4/4 time signature)
            self.current_bar = beat_count
            self.gui_queue.put(("current_beat", str(beat_count)))

            # Update the current chord display
            current_chord = self.chord_progression[self.current_chord_index]
            self.gui_queue.put(("current_chord", current_chord))

            time.sleep(self.metronome_interval)

    def update_tempo(self, *args):
        """
        Updates the metronome interval when the tempo changes.
        Ensures the tempo is always an integer.
        """
        tempo = int(float(self.tempo_var.get()))
        self.tempo_var.set(tempo)  # Ensure it's set as an integer
        self.metronome_interval = 60 / tempo

    def schedule_chord_change(self):
        """
        Schedules the next chord change based on the current tempo.
        """
        interval = self.metronome_interval * 4  # 4 beats per chord
        self.root.after(int(interval * 1000), self.update_chord)

    def toggle_channel1(self):
        self.channel1_enabled.set(not self.channel1_enabled.get())
        if not self.channel1_enabled.get():
            self.channel1_button.config(text="Enable 1", style="Red.TButton")
        else:
            self.channel1_button.config(text="Disable Channel 1", style="Green.TButton")

    def toggle_channel2(self):
        self.channel2_enabled.set(not self.channel2_enabled.get())
        if not self.channel2_enabled.get():
            self.channel2_button.config(text="Enable 2", style="Red.TButton")
            if self.note_on_sent_channel2 and self.last_notes_channel2:
                for note in self.last_notes_channel2:
                    midi_out.send(Message('note_off', note=note, channel=1))
                self.note_on_sent_channel2 = False
                self.channel2_velocity.set(0)
        else:
            self.channel2_button.config(text="Disable Channel 2", style="Green.TButton")

    def toggle_channel3(self):
        self.channel3_enabled.set(not self.channel3_enabled.get())
        if not self.channel3_enabled.get():
            self.channel3_button.config(text="Enable 3", style="Red.TButton")
            if self.note_on_sent_channel3 and self.last_notes_channel3:
                for note in self.last_notes_channel3:
                    midi_out.send(Message('note_off', note=note, channel=2))
                self.note_on_sent_channel3 = False
                self.channel3_velocity.set(0)
        else:
            self.channel3_button.config(text="Disable Channel 3", style="Green.TButton")

    def toggle_channel4(self):
        self.channel4_enabled.set(not self.channel4_enabled.get())
        if not self.channel4_enabled.get():
            self.channel4_button.config(text="Enable 4", style="Red.TButton")
            self.stop_midi_channel(3)  # Channel 4 is index 3
        else:
            self.channel4_button.config(text="Disable Channel 4", style="Green.TButton")

    def toggle_channel5(self):
        self.channel5_enabled.set(not self.channel5_enabled.get())
        if not self.channel5_enabled.get():
            self.channel5_button.config(text="Enable 5", style="Red.TButton")
            self.stop_midi_channel(4)  # Channel 5 is index 4
        else:
            self.channel5_button.config(text="Disable Channel 5", style="Green.TButton")

    def toggle_channel6(self):
        self.channel6_enabled.set(not self.channel6_enabled.get())
        if not self.channel6_enabled.get():
            self.channel6_button.config(text="Enable 6", style="Red.TButton")
            self.stop_midi_channel(5)  # Channel 6 is index 5
        else:
            self.channel6_button.config(text="Disable Channel 6", style="Green.TButton")

    def toggle_channel7(self):
        self.channel7_enabled.set(not self.channel7_enabled.get())
        if not self.channel7_enabled.get():
            self.channel7_button.config(text="Enable 7", style="Red.TButton")
            self.stop_midi_channel(6)  # Channel 7 is index 6
        else:
            self.channel7_button.config(text="Disable Channel 7", style="Green.TButton")

    def audio_process(self):
        # PyAudio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 2048

        p = pyaudio.PyAudio()

        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            return

        window = np.hamming(CHUNK)
        last_note = None
        hold_note = None  # This will hold the current note
        hold_note_time = 0  # Track how long we hold the note

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val != 0:
                audio_data /= max_val

            rms = np.sqrt(np.mean(audio_data**2))

            if rms < self.audio_silence_threshold_var.get():
                # Considered as silence, stop any playing note on Channel 1
                if last_note is not None:
                    midi_out.send(Message('note_off', note=last_note, channel=0))
                    last_note = None
                    hold_note = None
                self.gui_queue.put(("channel1_detected_pitch", 0.0))
                self.gui_queue.put(("channel1_corrected_pitch", "None"))
                continue

            audio_data = audio_data * window

            pitch_freq = yin_pitch_detection(
                audio_data, RATE, self.yin_threshold_var.get()
            )
            if pitch_freq is None:
                if last_note is not None:
                    midi_out.send(Message('note_off', note=last_note, channel=0))
                    last_note = None
                    hold_note = None
                self.gui_queue.put(("channel1_detected_pitch", 0.0))
                self.gui_queue.put(("channel1_corrected_pitch", "None"))
                continue

            midi_pitch = freq_to_midi(pitch_freq)
            if midi_pitch is None:
                continue

            self.gui_queue.put(("channel1_detected_pitch", round(pitch_freq, 2)))

            scale_notes = SCALES[self.selected_scale]
            scale_notes_midi = []
            for octave in range(-2, 3):
                for note in scale_notes:
                    midi_num = NOTE_TO_MIDI.get(note, None)
                    if midi_num is not None:
                        midi_num += (octave + self.octave_var_ch1.get() + 1) * 12
                        scale_notes_midi.append(midi_num)

            corrected_pitch = closest_note_in_scale(midi_pitch, scale_notes_midi)
            midi_note = int(round(corrected_pitch))

            corrected_freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
            corrected_note_name = self.get_note_name(midi_note)

            self.gui_queue.put(
                ("channel1_corrected_pitch", f"{corrected_note_name} ({midi_note})")
            )

            # If the new note is different from the held note, update the MIDI output
            if midi_note != hold_note:
                if hold_note is not None:
                    midi_out.send(Message('note_off', note=hold_note, channel=0))
                midi_out.send(
                    Message('note_on', note=midi_note, velocity=64, channel=0)
                )
                hold_note = midi_note
                hold_note_time = (
                    time.time()
                )  # Record the time when we start holding the note

            last_note = midi_note

            # Adjust how long you want to hold a note before sending another one (in seconds)
            if time.time() - hold_note_time > 2.0:  # 2 seconds for example
                hold_note_time = time.time()  # Reset the timer
                midi_out.send(Message('note_off', note=hold_note, channel=0))
                hold_note = None

        time.sleep(0.01)
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()

    def video_process(self):
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()  # Read a frame from the webcam
            if not ret:
                break
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]

            # YOLOv8 Pose detection
            results = self.yolo_model.detect_pose(frame)

            if results and len(results) > 0:
                keypoints_data = results[0]  # Assuming first detected person
                keypoints_xy = keypoints_data.xy
                keypoints_conf = keypoints_data.conf

                if keypoints_xy is not None and len(keypoints_xy) > 0:
                    if len(keypoints_xy[0]) > 10:
                        try:
                            right_wrist = keypoints_xy[0][10].cpu().numpy()
                            left_wrist = keypoints_xy[0][9].cpu().numpy()

                            if (
                                not (right_wrist == [0.0, 0.0]).all()
                                and not (left_wrist == [0.0, 0.0]).all()
                            ):
                                right_hand_pos_x, right_hand_pos_y = float(
                                    right_wrist[0]
                                ), float(right_wrist[1])
                                left_hand_pos_x, left_hand_pos_y = float(
                                    left_wrist[0]
                                ), float(left_wrist[1])

                                # Draw circles for wrist positions
                                cv2.circle(
                                    frame,
                                    (int(right_hand_pos_x), int(right_hand_pos_y)),
                                    5,
                                    (0, 0, 255),
                                    -1,
                                )
                                cv2.circle(
                                    frame,
                                    (int(left_hand_pos_x), int(left_hand_pos_y)),
                                    5,
                                    (255, 0, 0),
                                    -1,
                                )

                                if self.channel2_enabled.get():
                                    self.update_hand_control(
                                        right_hand_pos_x,
                                        right_hand_pos_y,
                                        "right",
                                        time.time(),
                                    )
                                if self.channel3_enabled.get():
                                    self.update_hand_control(
                                        left_hand_pos_x,
                                        left_hand_pos_y,
                                        "left",
                                        time.time(),
                                    )

                        except IndexError:
                            pass

            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Put the frame in the queue for GUI update
            self.gui_queue.put(("update_webcam", frame_rgb))

            time.sleep(0.01)

        cap.release()

    def run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def update_webcam_feed(self, frame):
        # Resize the frame to match the hand canvases (300x300)
        resized_frame = cv2.resize(frame, (300, 300))

        # Convert the frame to a PhotoImage
        image = Image.fromarray(resized_frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update the canvas with the new image
        self.webcam_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.webcam_canvas.image = (
            photo  # Keep a reference to prevent garbage collection
        )

    def calibrate_hand_range(self):
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.left_min_x, self.left_max_x = float('inf'), float('-inf')
        self.left_min_y, self.left_max_y = float('inf'), float('-inf')
        self.right_min_x, self.right_max_x = float('inf'), float('-inf')
        self.right_min_y, self.right_max_y = float('inf'), float('-inf')
        self.calibrate_button.config(text="Calibrating...", state="disabled")
        self.root.after(100, self.check_calibration)

    def check_calibration(self):
        if time.time() - self.calibration_start_time < self.calibration_duration:
            self.root.after(100, self.check_calibration)
        else:
            self.is_calibrating = False
            self.calibrate_button.config(text="Calibrate Hand Range", state="normal")
            print(f"Calibration complete.")

    def update_hand_control(self, pos_x, pos_y, hand_type, current_time):
        # Update observed range during calibration
        if self.is_calibrating:
            if hand_type == "left":
                self.left_min_x = min(self.left_min_x, pos_x)
                self.left_max_x = max(self.left_max_x, pos_x)
                self.left_min_y = min(self.left_min_y, pos_y)
                self.left_max_y = max(self.left_max_y, pos_y)
            else:  # right hand
                self.right_min_x = min(self.right_min_x, pos_x)
                self.right_max_x = max(self.right_max_x, pos_x)
                self.right_min_y = min(self.right_min_y, pos_y)
                self.right_max_y = max(self.right_max_y, pos_y)

        # Normalize based on observed range for each hand
        if hand_type == "left":
            x_range = max(1, self.left_max_x - self.left_min_x)
            y_range = max(1, self.left_max_y - self.left_min_y)
            normalized_x = (pos_x - self.left_min_x) / x_range
            normalized_y = (pos_y - self.left_min_y) / y_range
        else:  # right hand
            x_range = max(1, self.right_max_x - self.right_min_x)
            y_range = max(1, self.right_max_y - self.right_min_y)
            normalized_x = (pos_x - self.right_min_x) / x_range
            normalized_y = (pos_y - self.right_min_y) / y_range

        # Ensure values are within [0, 1]
        normalized_x = max(0, min(1, normalized_x))
        normalized_y = max(0, min(1, normalized_y))

        # Apply moving average filter
        if hand_type == "left":
            self.left_hand_positions.append((normalized_x, normalized_y))
            if len(self.left_hand_positions) > self.max_positions:
                self.left_hand_positions.pop(0)
            smoothed_x, smoothed_y = np.mean(self.left_hand_positions, axis=0)
        else:  # right hand
            self.right_hand_positions.append((normalized_x, normalized_y))
            if len(self.right_hand_positions) > self.max_positions:
                self.right_hand_positions.pop(0)
            smoothed_x, smoothed_y = np.mean(self.right_hand_positions, axis=0)

        # Octave mapping (1-7)
        if hand_type == "left":
            octave = int(
                np.interp(smoothed_x, [0, 1], [1, 7])
            )  # Higher octaves to the right
        else:
            octave = int(
                np.interp(smoothed_x, [0, 1], [7, 1])
            )  # Higher octaves to the right

        # Velocity mapping (0-127)
        velocity = int(
            np.interp(smoothed_y, [0, 1], [127, 0])
        )  # Higher velocity when hand is up

        if hand_type == "right" and self.right_hand_control_enabled:
            self.octave_var_ch2.set(octave)
            self.channel2_velocity.set(velocity)
            self.gui_queue.put(("update_right_hand", (smoothed_x, smoothed_y)))

            # Send MIDI message for Channel 2
            self.async_send_midi_message(2, octave, velocity)

        elif hand_type == "left" and self.left_hand_control_enabled:
            self.octave_var_ch3.set(octave)
            self.channel3_velocity.set(velocity)
            self.gui_queue.put(("update_left_hand", (smoothed_x, smoothed_y)))

            # Send MIDI message for Channel 3
            self.async_send_midi_message(3, octave, velocity)

    def update_gui(self):
        try:
            while True:
                item = self.gui_queue.get_nowait()
                if item[0] == "channel1_detected_pitch":
                    self.channel1_detected_pitch.set(item[1])
                elif item[0] == "channel1_corrected_pitch":
                    self.channel1_corrected_pitch.set(item[1])
                elif item[0] == "current_beat":
                    self.current_beat_label.config(text=item[1])
                elif item[0] == "current_chord":
                    self.current_chord_var.set(item[1])
                elif item[0] == "update_right_hand":
                    self.update_hand_position(
                        self.right_hand_canvas,
                        self.right_hand_dot,
                        item[1][0],
                        item[1][1],
                    )
                elif item[0] == "update_left_hand":
                    self.update_hand_position(
                        self.left_hand_canvas,
                        self.left_hand_dot,
                        item[1][0],
                        item[1][1],
                    )
                elif item[0] == "update_webcam":
                    self.update_webcam_feed(item[1])
        except queue.Empty:
            pass
        self.root.after(
            50, self.update_gui
        )  # Update more frequently for smoother visuals

    def on_closing(self):
        """Handle the window closing event"""
        self.running = False
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.quit()
        self.root.destroy()

    def run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def schedule_chord_change(self):
        """
        Schedules the next chord change based on the current tempo.
        """
        interval = self.metronome_interval * 4  # 4 beats per chord
        self.root.after(int(interval * 1000), self.update_chord)

    def update_chord(self):
        """
        Updates the current chord and schedules the next chord change.
        """
        # Generate new patterns for guitar and piano
        self.guitar_pattern = self.generate_guitar_pattern()
        self.piano_pattern = self.generate_piano_pattern()

        # Update chord index
        self.current_chord_index = (self.current_chord_index + 1) % len(
            self.chord_progression
        )
        # Update current chord display
        current_chord = self.chord_progression[self.current_chord_index]
        self.gui_queue.put(("current_chord", current_chord))

        # Update current chord notes for each channel
        for channel in range(1, 8):  # Channels 1-7
            setattr(
                self,
                f'current_chord_notes_channel{channel}',
                self.get_current_chord_notes(
                    getattr(self, f'octave_var_ch{channel}').get()
                ),
            )

        # Update current chord display
        self.current_chord_label.set(self.chord_progression[self.current_chord_index])

        # Schedule MIDI messages for each channel
        for channel in range(1, 8):  # Channels 1-7
            if getattr(self, f'channel{channel}_enabled').get():
                if channel == 1:
                    # For channel 1, we don't send MIDI messages here as it's handled in the audio_process method
                    continue
                velocity = (
                    getattr(self, f'channel{channel}_velocity').get()
                    if channel in [2, 3]
                    else 64
                )
                octave = getattr(self, f'octave_var_ch{channel}').get()

                if channel in [6, 7]:  # Guitar and Piano channels
                    pattern = (
                        self.guitar_pattern if channel == 6 else self.piano_pattern
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.async_play_pattern(
                            channel - 1,
                            self.get_current_chord_notes(octave),
                            pattern,
                            velocity,
                        ),
                        self.loop,
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        self.async_send_midi_message(channel, octave, velocity),
                        self.loop,
                    )

        # Update channel info for guitar and piano
        guitar_info = f"Playing pattern: {' '.join(self.guitar_pattern)}"
        piano_info = f"Playing pattern: {' '.join(self.piano_pattern)}"
        self.channel_info[6].set(f"Channel 6 (Guitar): {guitar_info}")
        self.channel_info[7].set(f"Channel 7 (Piano): {piano_info}")

        # Schedule the next chord change
        self.schedule_chord_change()

    async def async_play_pattern(self, channel_index, chord_notes, pattern, velocity):
        """Play a strum pattern for guitar or piano asynchronously"""
        for i, action in enumerate(pattern):
            if action == 'rest':
                await asyncio.sleep(
                    self.metronome_interval / (8 if channel_index == 5 else 4)
                )
                continue

            if channel_index == 5:  # Guitar
                if action == 'down':
                    for note in chord_notes:
                        midi_out.send(
                            Message(
                                'note_on',
                                note=note,
                                velocity=velocity,
                                channel=channel_index,
                            )
                        )
                        await asyncio.sleep(
                            0.01
                        )  # Slight delay between notes for more realistic strumming
                elif action == 'up':
                    for note in reversed(chord_notes):
                        midi_out.send(
                            Message(
                                'note_on',
                                note=note,
                                velocity=velocity,
                                channel=channel_index,
                            )
                        )
                        await asyncio.sleep(0.01)
            elif channel_index == 6:  # Piano
                if action == 'block':
                    for note in chord_notes:
                        midi_out.send(
                            Message(
                                'note_on',
                                note=note,
                                velocity=velocity,
                                channel=channel_index,
                            )
                        )
                elif action == 'arpeggio':
                    for note in chord_notes:
                        midi_out.send(
                            Message(
                                'note_on',
                                note=note,
                                velocity=velocity,
                                channel=channel_index,
                            )
                        )
                        await asyncio.sleep(0.05)  # Slight delay for arpeggio effect

            # Schedule note-off messages
            for note in chord_notes:
                self.loop.call_later(
                    self.metronome_interval / (8 if channel_index == 5 else 4),
                    lambda n=note: midi_out.send(
                        Message('note_off', note=n, velocity=0, channel=channel_index)
                    ),
                )

            # Wait for the next beat
            await asyncio.sleep(
                self.metronome_interval / (8 if channel_index == 5 else 4)
            )

        # Update the channel info after playing the pattern
        channel_name = "Guitar" if channel_index == 5 else "Piano"
        self.channel_info[channel_index + 1].set(
            f"Channel {channel_index + 1} ({channel_name}): Playing {' '.join(pattern)}"
        )

    async def async_send_midi_message(self, channel, octave, velocity):
        """
        Sends MIDI messages for the specified channel asynchronously.
        """
        channel_index = channel - 1  # Adjust for 0-based indexing
        current_chord_notes = self.get_current_chord_notes(octave)

        if getattr(self, f'send_chords_ch{channel}_var').get():
            # Send full chord
            for note in current_chord_notes:
                if getattr(
                    self, f'note_on_sent_channel{channel}', False
                ) and note in getattr(self, f'last_notes_channel{channel}', []):
                    midi_out.send(
                        Message(
                            'note_off', note=note, velocity=0, channel=channel_index
                        )
                    )
                midi_out.send(
                    Message(
                        'note_on', note=note, velocity=velocity, channel=channel_index
                    )
                )

            setattr(self, f'last_notes_channel{channel}', current_chord_notes.copy())
        else:
            # Send root note only
            note = current_chord_notes[0]
            if getattr(self, f'note_on_sent_channel{channel}', False) and getattr(
                self, f'last_notes_channel{channel}', []
            ):
                midi_out.send(
                    Message(
                        'note_off',
                        note=getattr(self, f'last_notes_channel{channel}')[0],
                        velocity=0,
                        channel=channel_index,
                    )
                )
            midi_out.send(
                Message('note_on', note=note, velocity=velocity, channel=channel_index)
            )

            setattr(self, f'last_notes_channel{channel}', [note])

        setattr(self, f'note_on_sent_channel{channel}', True)

    def safe_exit(self):
        """Safely exit the application"""
        self.running = False
        self.exit_event.set()  # Signal all threads to stop
        self.root.after(100, self.check_threads_and_close)

    def check_threads_and_close(self):
        """Check if threads have stopped and close the application"""
        if self.audio_thread.is_alive() or self.video_thread.is_alive():
            # Threads are still running, check again after a short delay
            self.root.after(100, self.check_threads_and_close)
        else:
            # All threads have stopped, now we can safely close the application
            self.on_closing()

    def on_closing(self):
        """Handle the window closing event"""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AutoTuneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
