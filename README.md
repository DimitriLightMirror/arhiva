# arhivadoc.eu - Backend OCR pentru arhive notariale

Sistem local-first de digitalizare a documentelor notariale romanesti:
scanare -> preprocesare -> OCR (Tesseract `ron`) -> analiza de layout ->
corectare LLM -> clasificare si extragere de campuri -> arhivare cu PDF
cautabil si sidecar JSON.

## Arhitectura

```
  Avision AD345GN (FTP/SMB/Folder partajat)
        |                                   \
        v                                    v
+------------------+               +----------------------+
|  WATCH_FOLDER    |               |  POST /api/scan      |
|  (FolderWatcher) |               |  (upload REST)       |
+--------+---------+               +----------+-----------+
         |                                    |
         v                                    v
+---------------------------------------------------------+
|                     DocumentPipeline                    |
|                                                         |
|  1. Ingest      PDF/TIFF/PNG/JPEG -> pagini (PyMuPDF)   |
|  2. Preprocess  OpenCV: orientare, deskew, denoise,     |
|                 binarizare adaptiva                     |
|  3. OCR         Tesseract ron (fallback ron+eng),       |
|                 TSV: cuvinte + casete + incredere       |
|  4. Layout      regiuni TEXT / TABLE / IMAGE / PLAN     |
|                 (detectie morfologica OpenCV)           |
|  5. Agent 1     corectare text OCR (LLM, optional)      |
|  6. Agent 2     clasificare + campuri + balize (LLM)    |
|  7. Export      PDF cautabil (ocrmypdf sau fallback     |
|                 reportlab) + result.json                |
+---------------------------------------------------------+
         |
         v
+---------------------------------------------------------+
|  ARCHIVE_ROOT/<clasa_document>/<an>/<doc_id>/           |
|     original_<fisier>  searchable.pdf  result.json      |
+---------------------------------------------------------+

LLM providers (config .env):
  LLM_PROVIDER=ollama  -> Ollama local (implicit, offline)
  LLM_PROVIDER=openai  -> orice API compatibil OpenAI
  LLM_PROVIDER=none    -> fara LLM; corectarea se sare, clasificarea
                          foloseste euristici pe cuvinte-cheie
```

## Instalare pe server Ubuntu/Debian

### 1. Dependinte de sistem

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv tesseract-ocr tesseract-ocr-ron
# optional, pentru PDF/A complet conform:
sudo apt install -y ocrmypdf ghostscript
```

### 2. Aplicatia

```bash
cd /opt && sudo git clone <repo> arhiva && cd arhiva
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # editati dupa nevoi
```

### 3. Ollama (agent LLM local, optional dar recomandat)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
# verificare: curl http://localhost:11434/api/tags
```

Fara Ollama, setati `LLM_PROVIDER=none` in `.env` - pipeline-ul functioneaza
in continuare (fara corectare, clasificare euristica).

### 4. Rulare

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
# pagina de test: http://server:8000/
```

Pentru productie folositi systemd (exemplu de unit in sectiunea de mai jos)
sau `docker compose up -d` (imaginile includ tesseract-ron si ocrmypdf).

```ini
# /etc/systemd/system/arhivadoc.service
[Unit]
Description=arhivadoc.eu OCR backend
After=network.target

[Service]
WorkingDirectory=/opt/arhiva
ExecStart=/opt/arhiva/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
EnvironmentFile=/opt/arhiva/.env

[Install]
WantedBy=multi-user.target
```

### 5. Conectarea scanerului Avision AD345GN

Scanerul trimite fisierele in `WATCH_FOLDER` (implicit `./watch`):

1. Configurati scanerul sa salveze prin retea (SMB/FTP) intr-un folder de pe
   server, de ex. `/srv/scans`.
2. Montati/partajati acel folder si setati `WATCH_FOLDER=/srv/scans` in `.env`.
3. Watcher-ul preia automat fisierele noi (PDF/TIFF/JPEG), asteapta sa fie
   scrise complet, le proceseaza si apoi le sterge din folderul de intrare.
   Rezultatele ajung in `ARCHIVE_ROOT/<clasa>/<an>/<doc_id>/`.

Alternativ, operatorul poate incarca manual din pagina web (`/`).

## API

| Metoda | Cale | Descriere |
|--------|------|-----------|
| POST | `/api/scan` | Upload scan (multipart, campul `file`). Raspuns: `{job_id}` |
| GET | `/api/jobs` | Lista joburi |
| GET | `/api/jobs/{id}` | Stare si etapa curenta (`queued/running/done/error`) |
| GET | `/api/jobs/{id}/result` | JSON complet: pagini, regiuni, text corectat, clasificare, campuri |
| GET | `/api/jobs/{id}/pdf` | Descarcare PDF cautabil |
| GET | `/api/jobs/{id}/json` | Descarcare sidecar JSON |
| GET | `/api/jobs/{id}/pages/{n}.jpg` | Imaginea paginii preprocesate |
| GET | `/api/health` | Stare serviciu + disponibilitate Tesseract |

Etapele raportate in `/api/jobs/{id}`: `ingest`, `preprocess`, `ocr`,
`layout`, `correction`, `classification`, `export`, `done`.

## Nota despre PDF/A

- Cu `ocrmypdf` instalat, exportul foloseste `--output-type pdfa` (PDF/A-2b).
- Fara `ocrmypdf`, se genereaza un PDF cautabil printr-un strat de text
  invizibil (reportlab) peste imaginile paginilor. Fisierul este cautabil si
  copiabil, dar nu este validat strict PDF/A; pentru conformitate arhivistica
  instalati ocrmypdf.

## Teste

```bash
pip install pytest
pytest tests/ -v
```

Testele folosesc mock-uri pentru Tesseract si LLM, deci ruleaza si fara
binarele de sistem.

## Structura proiectului

```
app/
  config.py            setari (pydantic-settings, .env)
  preprocess.py        orientare, deskew, denoise, binarizare
  ocr_engine.py        rasterizare + Tesseract TSV
  layout.py            segmentare TEXT/TABLE/IMAGE/PLAN
  llm/client.py        abstractizare provideri (ollama/openai/none)
  agents/correction.py Agent 1 - corectare OCR
  agents/classification.py Agent 2 - clasificare, campuri, balize
  storage.py           PDF cautabil, sidecar JSON, rutare arhiva
  pipeline.py          orchestrare joburi + watcher folder
  main.py              FastAPI
  static/              pagina web de test (vanilla JS)
tests/                 teste smoke (mock-uri)
```

## Roadmap

- [ ] OCR dedicat pentru planuri cadastrale (extragere vectori, legenda)
- [ ] Detectie stampile/semnaturi cu model antrenat (YOLO)
- [ ] Indexare full-text in Elasticsearch/Meilisearch peste arhiva
- [ ] Interfata de cautare si corectie manuala asistata
- [ ] Suport multi-notariat: utilizatori, roluri, jurnal audit
- [ ] Validare PDF/A automata (veraPDF) inainte de arhivare
- [ ] Fine-tuning model romanesc pentru corectia OCR notariala
