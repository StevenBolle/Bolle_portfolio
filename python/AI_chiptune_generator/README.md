# AI Chiptune Generator

An AI-assisted music generation application that uses a Long Short-Term Memory (LSTM) neural network to generate original chiptune-style melodies and combines them with rule-based musical arrangement to produce complete multi-track MIDI compositions.
The project includes a Flask web application that allows users to configure musical properties such as style, key, chord progression, tempo, track length, and generation temperature. Generated tracks can be synthesized and played directly in the browser, visualized through a CRT-inspired waveform display, and saved as MIDI files.
This project was originally developed for an introductory Artificial Intelligence course and was later reorganized and refined for inclusion in my software development portfolio.

## Demo

<!-- Add a screenshot, animated GIF, or YouTube demonstration here. -->
The web interface provides controls for configuring a composition, generating a new track, listening to the result, and viewing information about the generated composition.

## Features

- LSTM-based melody generation
- Rule-based chord, bass, and percussion arrangement
- Multiple musical styles, keys, and chord progressions
- Adjustable tempo, track length, and generation temperature
- Multi-track MIDI output
- In-browser MIDI synthesis and playback
- Play, stop, and replay controls for generated tracks
- CRT-inspired real-time waveform visualization
- Composer log containing information about each generated composition
- Optional WAV and MP3 rendering using FluidSynth and ffmpeg
- Experimental pipeline for training an LSTM using external MIDI datasets

## How It Works

The application uses a hybrid approach that combines machine learning with rule-based music generation.
The primary LSTM model generates the melody of each composition. Generation is conditioned using musical and stylistic information supplied by the application. The resulting melody is then combined with rule-based harmony, bass, percussion, and arrangement logic to create a complete composition.
The completed composition is written as a multi-track MIDI file. The Flask backend makes the generated MIDI available to the web interface, where Tone.js and the Web Audio API allow the track to be synthesized and played directly in the browser.
This approach allows the neural network to contribute the less predictable melodic component while deterministic music rules provide additional structure to the finished track.

## Web Application

The Flask web application provides the primary interface for the generator.
Users can configure several properties before generating a track:

- Track title
- Musical style
- Console conditioning
- Musical key
- Chord progression
- Speaker mode
- Number of bars
- Tempo
- Generation temperature

After generation, the application displays a composer log and allows the resulting MIDI composition to be played directly in the browser. A CRT-inspired waveform visualizer responds to the synthesized output during playback.
Browser MIDI playback does not require FluidSynth, ffmpeg, or an external SoundFont.

## Project Structure

```text
AI_chiptune_generator/
├── app/
│   ├── chiptune_core.py
│   ├── web_app.py
│   └── templates/
│       └── index.html
├── audio_data/
│   └── .gitkeep
├── midi_data/
│   └── .gitkeep
├── models/
│   └── melody_lstm.pt
├── training/
│   └── real_midi/
│       ├── midi_melody_extract.py
│       ├── test_real_melody_model.py
│       └── train_real_melody_lstm.py
├── .gitignore
├── requirements.txt
└── README.md

### `app/`

Contains the primary application.
`chiptune_core.py` contains the neural-network model definitions, melody-generation logic, musical arrangement logic, MIDI generation, and supporting functions used to construct a complete track.
`web_app.py` provides the Flask backend, loads the trained model, handles generation requests, serves generated MIDI files, and optionally renders MIDI into conventional audio formats.
`templates/index.html` contains the browser interface, playback controls, and CRT-style waveform visualizer.

### `models/`

Contains the trained LSTM model used by the main application.

### `midi_data/`

Stores MIDI files and composer logs created while the application is running. Generated files are excluded from version control.

### `audio_data/`

Stores optional WAV and MP3 files produced by the external audio-rendering pipeline. Generated files are excluded from version control.

### `training/real_midi/`

Contains an experimental alternative training pipeline developed to investigate training the melody model using existing MIDI datasets.
This pipeline is separate from the trained model used by the main web application.

## Requirements

The primary application requires Python and the packages listed in `requirements.txt`.
The main Python dependencies are:

- Flask
- NumPy
- PyTorch
- Mido

The browser interface also uses Tone.js and `@tonejs/midi` for client-side MIDI synthesis and playback.

## Installation

Clone or download the repository and navigate to the project directory.
Create and activate a Python virtual environment if desired.
Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app/web_app.py
```

The development server will start locally. Open the address displayed by Flask in a web browser to use the generator.

## Generating a Track

1. Enter an optional track title.
2. Select the desired musical parameters.
3. Press **Generate**.
4. Wait for the LSTM model and arrangement system to create the composition.
5. Press **Listen** to synthesize and play the generated MIDI in the browser.
6. Press **Stop** to end playback early, or allow the composition to finish normally.

A composer log is generated alongside the MIDI output and displays information about the composition.
Because generation contains a stochastic component, repeated generations using similar settings can produce different melodies.

## Browser MIDI Playback

Generated compositions can be played directly in the browser without converting the MIDI file into WAV or MP3 audio.
The application uses `@tonejs/midi` to read the generated MIDI and Tone.js to synthesize its notes through the browser's Web Audio API. This also provides the audio data used by the CRT-style waveform visualizer.
This browser-based playback path makes the core application usable without requiring a locally installed SoundFont or external audio-rendering software.

## Optional Audio Rendering

The application also contains support for rendering generated MIDI files into WAV and MP3 audio.
This functionality requires additional software that is not included with the repository:

- FluidSynth
- ffmpeg
- A compatible SoundFont (`.sf2`)

When configured, FluidSynth renders the MIDI composition using the supplied SoundFont and ffmpeg can convert the resulting WAV file into MP3 format.
SoundFont files are intentionally excluded from this repository because they are third-party resources with their own licensing and distribution requirements.
The optional rendering pipeline is not required to generate or listen to tracks through the web application.

## Experimental Real-MIDI Training Pipeline

During development, I also investigated whether the melody model could be trained using an external collection of existing MIDI music.
The experimental pipeline consists of three stages:

1. `midi_melody_extract.py` extracts and normalizes likely melody sequences from MIDI files.
2. `train_real_melody_lstm.py` converts those sequences into training data and trains a separate LSTM model.
3. `test_real_melody_model.py` generates a melody from the experimental model and combines it with a rule-based arrangement for evaluation.

This experiment remained separate from the final web application. The primary application uses the synthetic and rule-conditioned LSTM model stored in `models/melody_lstm.pt`.
The MIDI corpus, extracted training sequences, and experimental model checkpoint are not distributed with this repository. The training scripts are retained to document the experimental approach and allow the pipeline to be studied or adapted using independently supplied MIDI data.

## Limitations

This project is an experimental AI music generator rather than a production music composition system.
Generated musical quality varies between tracks because melody generation is probabilistic. The rule-based arrangement system provides structure, but it does not guarantee that every generated melody will produce an equally successful composition.
The console setting is used as a conditioning input during generation, but generated MIDI does not emulate the original sound hardware of systems such as the NES, SNES, Genesis, or Game Boy. Audible timbre ultimately depends on the synthesizer or SoundFont used to render the MIDI.
The real-MIDI training experiment was not integrated into the final application and is included as a record of an alternative approach explored during development.

## Technologies

- Python
- PyTorch
- Flask
- NumPy
- Mido
- HTML
- CSS
- JavaScript
- Tone.js
- Web Audio API
- MIDI
- FluidSynth (optional)
- ffmpeg (optional)

## Project Background

This project was originally created as a final project for an Artificial Intelligence course as an exploration of neural-network-based music generation.
The project was not assigned with a rubric, we were tasked with utilizing our skills to create a project of our own choice that uses AI in an interesting manner.
With that freedom, I decided to mesh my love of chiptune music to the coursework. This project is the results of those efforts.
Development included designing and training an LSTM melody generator, creating a rule-based arrangement system, producing multi-track MIDI output, experimenting with training from existing MIDI data, and building a browser-based interface around the generation system.

The project was later reorganized for portfolio use. This included separating application, model, and experimental training components; removing training data and third-party assets from the repository; improving generated-audio playback; and adding direct browser-based MIDI synthesis so the application can demonstrate its primary functionality without requiring an external SoundFont.

## License and Third-Party Assets

This repository does not distribute commercial music datasets, third-party SoundFonts, or other externally sourced music assets used during experimentation.
Anyone using the experimental training pipeline should provide MIDI data they have the appropriate rights or permission to use.