import asyncio
import soundfile as sf
import pygame
import openai
import numpy as np
from datetime import datetime
import logging
from queue import Queue
import pyaudio
import io
import sys
import os
import platform
from scipy import signal
from elevenlabs import ElevenLabs
from openai import APIConnectionError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

logger.info(f"Looking for .env file at: {env_path}")
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(".env file found and loaded")
else:
    logger.error(".env file not found")

# Force UTF-8 encoding for console output - For other language support
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize pygame mixer with explicit sample rate
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    logger.info("Pygame mixer initialized with frequency=44100, size=-16, channels=1")
except Exception as e:
    logger.error(f"Failed to initialize pygame mixer: {e}")
    raise

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY not set")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, max_retries=5, timeout=30.0)

# Initialize ElevenLabs client
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY environment variable not set")
    raise ValueError("ELEVENLABS_API_KEY not set")
try:
    logger.info("Initializing ElevenLabs client...")
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs client: {e}")
    raise

# Global variables
chat_history = []
response_queue = Queue()
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
MIN_SPEECH_AMPLITUDE = 1000
TTS_CACHE = {}
MAX_CACHE_SIZE = 5
USE_LOCAL_WHISPER = False

async def transcribe_audio():
    """
    Records audio chunks until silence is detected, then transcribes using OpenAI Whisper.
    Returns the transcribed text prompt.
    """
    start_time = datetime.now()
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
        
        audio_chunks = []
        silence_time = 0
        chunk_duration = CHUNK_SIZE / SAMPLE_RATE

        logger.info("Recording audio... Speak now.")
        while silence_time < SILENCE_DURATION:
            chunk = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)
            audio_chunks.append(chunk)
            max_amplitude = np.max(np.abs(chunk))
            logger.debug(f"Chunk max amplitude: {max_amplitude}")
            if max_amplitude < SILENCE_THRESHOLD:
                silence_time += chunk_duration
            else:
                silence_time = 0

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.concatenate(audio_chunks)
        max_amplitude = np.max(np.abs(audio_data))
        logger.info(f"Audio buffer max amplitude: {max_amplitude}, length: {len(audio_data)} samples")
        logger.info(f"Recording took {(datetime.now() - start_time).total_seconds():.2f} seconds")

        if max_amplitude < MIN_SPEECH_AMPLITUDE:
            logger.info("Audio buffer too quiet, skipping transcription")
            return ""

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        logger.info(f"Buffer size: {buffer.getbuffer().nbytes} bytes")
        logger.info(f"Buffer creation took {(datetime.now() - start_time).total_seconds():.2f} seconds")

        debug_file = "debug_audio.wav"
        with open(debug_file, "wb") as f:
            f.write(buffer.getbuffer())

        if USE_LOCAL_WHISPER:
            pass
        else:
            for attempt in range(2):
                try:
                    with open(debug_file, "rb") as audio_file:
                        transcription_start = datetime.now()
                        transcription = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text",
                            language="en"
                        )
                        logger.info(f"API transcription took {(datetime.now() - transcription_start).total_seconds():.2f} seconds")
                    prompt = transcription.strip()
                    logger.info(f"Transcribed prompt: {prompt}")
                    logger.info(f"Total transcription took {(datetime.now() - start_time).total_seconds():.2f} seconds")
                    return prompt
                except APIConnectionError as api_conn_error:
                    logger.error(f"API connection error (attempt {attempt + 1}): {api_conn_error}")
                    await asyncio.sleep(2 ** attempt)
                except AuthenticationError as auth_error:
                    logger.error(f"Authentication error (attempt {attempt + 1}): {auth_error}")
                    return ""
                except RateLimitError as rate_error:
                    logger.error(f"Rate limit error (attempt {attempt + 1}): {rate_error}")
                    await asyncio.sleep(2 ** attempt)
                except Exception as api_error:
                    logger.error(f"Unexpected API error (attempt {attempt + 1}): {type(api_error).__name__}: {api_error}")
                    await asyncio.sleep(2 ** attempt)
            logger.error("All API attempts failed")
            return ""
    except Exception as e:
        logger.error(f"Transcription error: {type(e).__name__}: {e}")
        return ""

async def adjust_response(prompt, response_text):
    """
    This could be used for adding emotion to responses, by adjustin the tone based on what emotions does AI detect
    """
    response_text = f"{response_text}"
    return response_text

async def store_in_memory(prompt, response):
    """
    Store conversation in memory (chat_history) with role and content.
    """
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": response})

async def generate_response(prompt):
    """
    Generate a response using OpenAI's GPT-4 model.
    """
    start_time = datetime.now()
    messages = [
        {"role": "system", "content": "You're an AI assistant"}
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.5
        )
        response_text = response.choices[0].message.content.strip()
        response_text = await adjust_response(prompt, response_text)
        await store_in_memory(prompt, response_text)
        logger.info(f"Response generated: {response_text}")
        logger.debug(f"Response encoding: {response_text.encode('utf-8')}")
        logger.info(f"Response generation took {(datetime.now() - start_time).total_seconds():.2f} seconds")
        response_queue.put_nowait(response_text)
        return response_text
    except Exception as e:
        logger.error(f"Response generation error: {type(e).__name__}: {e}")
        return ""

def write_response_to_file(text):
    """
    Convert text to speech using ElevenLabs API and save to in-memory buffer.
    Returns the buffer for playback.
    """
    start_time = datetime.now()
    try:
        if text in TTS_CACHE:
            logger.info("Using cached TTS audio")
            return TTS_CACHE[text]

        if len(TTS_CACHE) >= MAX_CACHE_SIZE:
            TTS_CACHE.pop(next(iter(TTS_CACHE)))

        generation_start = datetime.now()
        audio_stream = elevenlabs_client.generate(
            text=text,
            voice="Rachel",
            model="eleven_monolingual_v1",
            output_format="pcm_16000"
        )
        audio_data = b''.join(chunk for chunk in audio_stream)
        logger.info(f"TTS generation took {(datetime.now() - generation_start).total_seconds():.2f} seconds")
        logger.info(f"TTS audio data size: {len(audio_data)} bytes")

        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767
        sample_rate = 16000

        logger.info(f"TTS output: sample_rate={sample_rate} Hz, shape={audio_array.shape}, dtype={audio_array.dtype}, range=[{audio_array.min()}, {audio_array.max()}]")

        if sample_rate != SAMPLE_RATE:
            logger.info(f"Resampling audio from {sample_rate} Hz to {SAMPLE_RATE} Hz")
            num_samples = int(len(audio_array) * SAMPLE_RATE / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)
            sample_rate = SAMPLE_RATE

        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()

        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        debug_tts_file = f"debug_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with open(debug_tts_file, "wb") as f:
            f.write(buffer.getvalue())
        logger.info(f"Saved TTS output to {debug_tts_file}")

        buffer_copy = io.BytesIO(buffer.getvalue())
        TTS_CACHE[text] = buffer_copy
        logger.info(f"Total TTS processing took {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return buffer
    except Exception as e:
        logger.error(f"Error writing response to file: {e}")
        return None

def audio_playback(audio_buffer):
    """
    Play the audio from the in-memory buffer using pygame.
    """
    try:
        start_time = datetime.now()
        logger.info("Starting audio playback")
        sound = pygame.mixer.Sound(audio_buffer)
        sound.play()
        while pygame.mixer.get_busy():
            pygame.time.wait(100)
        logger.info(f"Audio playback completed, took {(datetime.now() - start_time).total_seconds():.2f} seconds")
    except Exception as e:
        logger.error(f"Audio playback error: {e}")

async def main():
    while True:
        prompt = await transcribe_audio()
        if prompt:
            response_text = await generate_response(prompt)
            if response_text:
                audio_buffer = write_response_to_file(response_text)
                if audio_buffer:
                    audio_playback(audio_buffer)
        await asyncio.sleep(0.1)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())