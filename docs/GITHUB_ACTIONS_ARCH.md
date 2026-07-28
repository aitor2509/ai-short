# Arquitectura cloud 100% gratis (sin tarjeta, sin PC propia)

## El ejecutor: GitHub Actions

Es el **único servicio cloud** que te da procesamiento real (2 CPUs, 7 GB RAM, 2,000 min/mes en repo privado o **ilimitado en repo público**) sin pedir tarjeta de crédito.

| Recurso | GitHub Actions (público) |
|---|---|
| CPUs | 2 vCPU |
| RAM | 7 GB |
| Disco | 14 GB SSD |
| Minutos | **Ilimitados** en repo público |
| Paralelismo | Hasta 20 jobs simultáneos |
| Límite por workflow | 6 h de ejecución |
| Costo | **$0** |
| Tarjeta requerida | No |

## Arquitectura

```
GitHub Actions (cloud, 24/7 con cron)
         │
    ┌────┴────────────────────────────┐
    │   workflow orquestador          │
    │   (se ejecuta cada hora, 24/7)  │
    └────┬────────────────────────────┘
         │
    ┌────┴────────────────────────────┐
    │  Jobs paralelos (hasta 20)      │
    │                                  │
    │  job-1: video_A                  │
    │  job-2: video_B                  │
    │  ...                             │
    │  job-20: video_T                 │
    │                                  │
    │  Cada job hace:                  │
    │   1. yt-dlp download (~2 min)    │
    │   2. Groq Whisper || whisper CPU │
    │   3. signals.py (heatmap+audio)  │
    │   4. Gemini LLM (1 llamada)      │
    │   5. ffmpeg clip (~1 min)        │
    │   6. Guarda como artifact        │
    └──────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────┐
    │  workflow publicador (cada 30m)  │
    │  Saca clips de artifacts         │
    │  Sube a Facebook (Graph API)     │
    │  200 calls/hora por app          │
    └──────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────┐
    │  GitHub Pages (storage gratis)   │
    │  Los clips subidos sirven como   │
    │  URL para Facebook (file_url)    │
    └──────────────────────────────────┘
```

## Flujo completo

### 1. Contenedor de estado: GitHub Issues o JSON en el repo

No necesitas base de datos. Guardas el estado como JSON en el propio repo (se hace commit automático):

```json
{
  "accounts": [
    {
      "name": "minecraft_es",
      "niche": "minecraft",
      "facebook_page_id": "xxx",
      "facebook_token": "xxx",
      "sources": ["@Minecraft", "@Dream"],
      "schedule": ["08:00", "14:00", "20:00"]
    }
  ],
  "queue": [
    {
      "video_url": "https://youtu.be/...",
      "account": "minecraft_es",
      "status": "pending"
    }
  ],
  "published_today": 83
}
```

### 2. Workflow #1: Content Discovery (cada 6h)

```yaml
name: descubrimiento
on:
  schedule:
    - cron: '0 */6 * * *'  # cada 6 horas
  workflow_dispatch:

jobs:
  discover:
    runs-on: ubuntu-latest
    steps:
      - usa tu trend_finder.py para 20 cuentas
      - compara con queue actual (evita duplicados)
      - actualiza queue.json
      - hace commit
```

### 3. Workflow #2: Procesamiento (cada hora)

```yaml
name: procesar
on:
  schedule:
    - cron: '0 * * * *'  # cada hora
  workflow_dispatch:

jobs:
  process:
    # Máquina matrix: procesa hasta 20 videos en paralelo
    strategy:
      matrix:
        job: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    steps:
      - leer queue.json
      - tomar siguiente video pendiente
      - yt-dlp download
      - if Groq quota disponible → transcribe con Groq (segundos)
        else → faster-whisper local en CPU (~5-10 min)
      - signals.py (heatmap + audio + comments)
      - Gemini API (1 llamada, veredicto)
      - ffmpeg clip + reframe 9:16
      - subir a GitHub Pages como artifact
      - marcar como "done" en queue.json
      - commit
```

### 4. Workflow #3: Publicación (cada 30 min)

```yaml
name: publicar
on:
  schedule:
    - cron: '*/30 * * * *'
  workflow_dispatch:

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - leer queue.json
      - por cada clip "done" y cuenta con horario:
        - POST /{page_id}/videos a Facebook Graph API
        - marcar como "published"
        - commit
```

## Tiempos estimados

Con 20 jobs en paralelo:

```
Batch de 20 videos:
  yt-dlp:         ~2 min  (paralelo)
  transcripción:  ~5 min  (mix Groq + CPU whisper)
  señales:        ~1 min
  Gemini LLM:     ~5 seg
  ffmpeg clip:    ~1 min
  ─────────────────────
  Total por batch: ~7-10 min

Por hora:    4-5 batches × 20 = 80-100 videos/hora
En 2 horas:  120-200 videos procesados
```

## APIs gratuitas integradas

| Servicio | Uso | Límite gratis |
|---|---|---|
| **Groq Whisper** | Transcripción rápida | ~20-50 requests/día |
| **Gemini 1.5 Flash** | LLM veredicto | 1,500 requests/día |
| **Facebook Graph API** | Publicación | ~200 calls/hora |
| **GitHub Actions** | Cómputo | Ilimitado (repo público) |
| **GitHub Pages** | Host de videos | 1 GB, 100 GB ancho de banda/mes |

## Cosas a tener en cuenta

### GitHub Actions en repos públicos
- Cualquiera puede ver los workflows, logs y artifacts
- Solución: **repo privado** → 2,000 min/mes ≈ ~200 videos/mes (no alcanza)
- O **repo público con los videos subidos a GitHub Pages** (público pero igual cualquiera ve los clips si están publicados en Facebook)
- Alternativa: usar un repo privado y rotar meses (un mes procesas, otro publicas)

### Ancho de banda de GitHub Pages (100 GB/mes)
- Cada short vertical: ~5-10 MB
- 120 videos/día × 30 días = 3,600 videos/mes
- 3,600 × 7 MB = **~25 GB/mes** → ✅ dentro del límite
- Facebook descarga 1 vez por video

### Si GitHub Actions te sabe a poco

Sigue siendo la opción más potente sin tarjeta. Si más adelante consigues tarjeta, migras a:

```
GitHub Actions (orquestación)
    │
    └──▶ RunPod / Vast.ai (GPU, $0.18/h, pago con PayPal/cripto)
         Transcripción GPU: 1h audio → 30 segundos
         Costo 120 videos/día: ~$8-10/mes
```

## TL;DR

**¿Se puede 100% gratis en cloud sin tarjeta?** Sí, con GitHub Actions (repo público) + Groq + Gemini + Facebook Graph API. Procesas 120 videos/día en ~2-3 horas con 20 workers paralelos. Publicas a Facebook sin límites.

**¿Peaje?** El repo tiene que ser público para los minutos ilimitados (los logs y artifacts son visibles). Si eso no te importa, la arquitectura funciona.
