# Contexto del proyecto
Pipeline en Python que genera shorts verticales desde videos largos de YouTube,
para republicación en cuentas con licencia/permiso del creador original.

## Stack
- yt-dlp (descarga)
- faster-whisper local / Groq Whisper API (transcripción)
- ffmpeg + OpenCV (recorte, reencuadre 9:16)
- Detección de highlights: heatmap "Most Replayed" + picos de audio + comentarios con timestamp + scene change + motion
- LLM (Gemini -> Groq -> LM Studio local, cascada solo ante error de cuota) para veredicto final
- **Editor mode** (`mode=editor`): señales ANTES de transcripción, transcribe solo ventanas candidatas (~10 min/video en vez de ~67 min)
- GitHub Actions para ejecución cloud gratuita (repos públicos, minutos ilimitados)

## Reglas
- Nunca hardcodear API keys, usar variables de entorno (.env)
- Loguear cada etapa del pipeline (descarga, transcripción, detección, corte)
- Manejar rate limits explícitamente, con backoff y cambio de proveedor solo en error 429/quota
- El prefiltro sin IA debe ejecutarse SIEMPRE antes de llamar al LLM (para minimizar llamadas)

## Cómo probar
- python -m pytest tests/
- python editor.py "https://youtu.be/..." --count 10 --output test_clips  (editor local)
- python editor.py --batch urls.txt --count 50                          (procesar lote)
- El workflow editor.yml via GitHub Actions (workflow_dispatch con URLs)
