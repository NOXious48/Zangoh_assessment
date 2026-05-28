import asyncio
import edge_tts
import os

# The 4 test queries covering different topics in the knowledge base
queries = [
    ("test.mp3", "Testing the speech to text system. One, two, three."),
    ("question.mp3", "What is your return policy for international orders?"),
    ("shipping.mp3", "How long does standard shipping take?"),
    ("warranty.mp3", "My electronics are broken. What is covered under the warranty?")
]

async def create_audio():
    print("Generating 4 test audio files...")
    for filename, text in queries:
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(filename)
        print(f"Created {filename} - Content: '{text}'")
    print("All files generated successfully!")

if __name__ == "__main__":
    asyncio.run(create_audio())
