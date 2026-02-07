"""Simple test script for the drum transcription API."""

import asyncio
import httpx
import sys
from pathlib import Path

async def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8002"
    
    async with httpx.AsyncClient() as client:
        # Test health endpoint
        print("Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"Health status: {response.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return
        
        # Test root endpoint
        print("\nTesting root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"API info: {response.json()}")
        except Exception as e:
            print(f"Root endpoint failed: {e}")
        
        # Test transcription (if audio file provided)
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            if Path(audio_file).exists():
                print(f"\nTesting transcription with {audio_file}...")
                try:
                    with open(audio_file, "rb") as f:
                        files = {"file": (audio_file, f, "audio/mpeg")}
                        data = {
                            "threshold": 0.5,
                            "min_interval": 0.05,
                            "tempo": 120,
                            "use_alternative_notes": False
                        }
                        response = await client.post(
                            f"{base_url}/transcribe",
                            files=files,
                            data=data
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"Transcription successful!")
                        print(f"Statistics: {result.get('statistics', {})}")
                        
                        # Download MIDI file
                        if result.get('midi_file_url'):
                            midi_url = f"{base_url}{result['midi_file_url']}"
                            midi_response = await client.get(midi_url)
                            if midi_response.status_code == 200:
                                output_file = f"output_{Path(audio_file).stem}.mid"
                                with open(output_file, "wb") as f:
                                    f.write(midi_response.content)
                                print(f"MIDI file saved as: {output_file}")
                    else:
                        print(f"Transcription failed: {response.text}")
                        
                except Exception as e:
                    print(f"Transcription test failed: {e}")
            else:
                print(f"Audio file not found: {audio_file}")
        else:
            print("\nTo test transcription, provide an audio file path:")
            print("python test_api.py /path/to/your/audio.mp3")

if __name__ == "__main__":
    asyncio.run(test_api())
