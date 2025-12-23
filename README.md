# NLP Pipeline (AWS/Local) - Sentiment, Key Phrases, NER y Traducción (opcional)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square)
![AWS](https://img.shields.io/badge/AWS-Comprehend%20%7C%20Translate-232F3E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

Este proyecto implementa un **pipeline de NLP reproducible** que puede ejecutarse en:

- **AWS**: Amazon Comprehend (sentiment, key phrases, entities) y Amazon Translate (opcional).
- **Localmente**: VADER (sentiment), TF‑IDF (key phrases) y **candidatos heurísticos** de entidades vía regex.
- **Traducción local opcional**: HuggingFace MarianMT (descarga el modelo la primera vez).

El notebook incluye logs exportables, control de costo/tiempo y outputs en CSV/JSON.

---

## Objetivos

Dado un dataset en CSV con una columna de texto, el pipeline:

1. Extrae y normaliza textos (limpieza mínima + truncado).
2. Genera features TF‑IDF (para modo local).
3. Calcula **sentiment** (AWS o VADER local).
4. Extrae **key phrases** (AWS o TF‑IDF local).
5. Extrae **entidades**:
   - AWS: NER real con Comprehend
   - Local: *entity candidates* por regex 
6. (Opcional) Traduce INGLES --> ESPAÑOL (AWS Translate o HuggingFace local).

---

## Notebook

- `nlp_pipeline_aws_local.ipynb`

---

## Outputs

Se generan en `outputs/`:

- `pipeline_results.csv` / `pipeline_results.json` — dataset final consolidado (1 fila por documento)
- `keyphrases_entities.csv` — key phrases y entidades/candidatos por documento
- `translations_en_es.csv` — traducciones INGLES --> ESPAÑOL (si se activa)
- `pipeline_logs.json` — logs del pipeline (eventos + metadata)

---

## Requisitos

- Python **3.10+**
- JupyterLab / Jupyter Notebook
- (Opcional) Credenciales AWS con permisos para Comprehend/Translate

Instalación local:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

## Costos al usar AWS (IMPORTANTE)

Este proyecto puede ejecutarse en **modo local** o en **AWS**.  
Si lo ejecutas en AWS, algunos servicios pueden generar costos reales (especialmente si procesas muchos documentos o dejas recursos encendidos).

### Servicios que pueden generar costo
- **Amazon Comprehend**: cobra por análisis de texto.
- **Amazon Translate**: cobra por caracteres traducidos.
- **SageMaker / EC2**: instancias de notebook/compute si se quedan ejecutándose.
- (Opcional) otros recursos del entorno AWS.

> Recomendación: corre en AWS solo si necesitas demostrar integración con servicios administrados.  
> Para pruebas rápidas o demos, usa el **modo local**.

### Cómo minimizar costos (recomendado)
1. **Mantén la traducción desactivada** (default):
   - `ENABLE_TRANSLATE=0`
2. **Procesa menos documentos** en pruebas:
   - `MAX_DOCS=500` (o menos)
   - `CSV_NROWS=5000`
   - `MAX_TRANSLATE_DOCS=50` si activas traducción
3. **Forzar modo local (costo 0)**:
   - `NLP_PROVIDER=local`
4. **Si usas SageMaker/Studio**: apaga el kernel/instancia cuando termines.
5. **Activa alertas**:
   - AWS Budgets / Cost Explorer / Cost Anomaly Detection (si tu cuenta lo permite).

### En ejecuciones sin límites los costos pueden subir rápidamente.

[![Costos-de-uso-aws.jpg](https://i.postimg.cc/JnM3rJt8/Costos-de-uso-aws.jpg)](https://postimg.cc/wt4RF1h4)

---

## Cómo ejecutar

### Opción A - Localmente

**PowerShell**
```powershell
$env:NLP_PROVIDER="local"
jupyter lab
```

**Mac/Linux**
```bash
export NLP_PROVIDER=local
jupyter lab
```

>Abre el notebook y ejecuta **Run All**.

---

### Opción B - AWS (Comprehend / Translate)

Requiere credenciales configuradas y permisos IAM para:

- `comprehend:BatchDetectSentiment`
- `comprehend:BatchDetectKeyPhrases`
- `comprehend:BatchDetectEntities`
- `translate:TranslateText` (si activas traducción)

Config:

```bash
export NLP_PROVIDER=aws
export AWS_REGION=us-east-1
```

> Si estás en SageMaker/Studio, el rol de ejecución debe tener permisos explícitos (si no, usa `NLP_PROVIDER=local` en celda 2, por defecto en 'auto').

---

## Dataset (CSV) 

El pipeline funciona con **cualquier CSV** que tenga una columna con texto.
Puedes probar a usar el que esta por defecto o usar otro

Por defecto:
- `DATA_PATH="data/AMAZON-REVIEW-DATA-CLASSIFICATION.csv"`
- `TEXT_COLUMN="reviewText"`

Para usar otro dataset:

**PowerShell**
```powershell
$env:DATA_PATH="data/mi_dataset.csv"
$env:TEXT_COLUMN="text"
```

**Mac/Linux**
```bash
export DATA_PATH="data/mi_dataset.csv"
export TEXT_COLUMN="text"
```

**Notebook - Celda 2**
```bash
# Cambia directamente en el notebook de esta forma
DATA_PATH = Path(os.getenv("DATA_PATH", "data/EJEMPLO_DE_DATASET.csv"))
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "NOMBRE_DE_COLUMNA_CON_TEXTO")
```

Notas:
- Si tu CSV usa `;` como separador, ajusta la lectura en la **CELDA 4** (`sep=";"`).
- Si tu CSV no es UTF‑8, cambia `encoding` en **CELDA 4** (por ejemplo `latin-1` / `cp1252`).
- Algunos datasets públicos traen filas corruptas; si aparece `ParserError` / `unexpected end of data`, usa `engine="python"` + `on_bad_lines="skip"` en la carga.

---

## Parámetros configurables (variables de entorno)

Estos valores controlan costo/tiempo sin tocar el código:

- **Provider**
  - `NLP_PROVIDER` = `auto` | `aws` | `local`
  - `AWS_REGION` (default: `us-east-1`)

- **Tamaño de ejecución**
  - `MAX_DOCS` (default: `5000`) — documentos a procesar
  - `CSV_NROWS` (default: `100000`) — filas máximas a leer del CSV
  - `RAW_TEXT_MAX_CHARS` (default: `1000`) — truncado por documento

- **TF‑IDF**
  - `TFIDF_MAX_FEATURES` (default: `5000`) — vocabulario máximo

- **NER en AWS**
  - `AWS_COMPREHEND_BATCH` (default: `25`)
  - `AWS_ENTITY_MIN_SCORE` (default: `0.80`)

- **Traducción (opcional)**
  - `ENABLE_TRANSLATE` (default: `0`) — 0/1 para activar
  - `MAX_TRANSLATE_DOCS` (default: `200`) — documentos a traducir
  - `HF_TRANSLATE_MODEL` (default: `Helsinki-NLP/opus-mt-en-es`)
  - `HF_HOME` (default: `.hf_cache`)
  - `HF_BATCH_SIZE` (default: `16`)
  - `HF_MAX_LENGTH` (default: `128`)

---

## Activar traducción INGLES --> ESPAÑOL (opcional)

>Por defecto esta opción viene desactivada.

**PowerShell**
```powershell
$env:ENABLE_TRANSLATE="1"
$env:MAX_TRANSLATE_DOCS="200"
```

**Mac/Linux**
```bash
export ENABLE_TRANSLATE=1
export MAX_TRANSLATE_DOCS=200
```

**Notebook - Celda 2**
```bash
ENABLE_TRANSLATE = os.getenv("ENABLE_TRANSLATE", "0") == "1" # Cambia el valor en ("ENABLE_TRANSLATE", "0"), por defecto esta en 0
MAX_TRANSLATE_DOCS = int(os.getenv("MAX_TRANSLATE_DOCS", "200")) # Cambia este valor para controlar la cantidad de documentos a traducir
```

- Si el servicio de AMAZON Translate esta disponible --> usa AWS (puede generar costos).
- Si el servicio de AMAZON Translate no esta disponible --> usa HuggingFace local (consume CPU/RAM).

---

## Notas de interpretación 

- En **modo local**, `conf_pos/conf_neg` vienen de VADER (`compound`) y son **proxies**, no probabilidades como AWS Comprehend.
- En **modo local**, `entity_candidates_local` son **candidatos heurísticos** (regex), **no NER real**.

---

## Troubleshooting rápido

- **NLTK LookupError (punkt/stopwords)**: ejecuta las celdas iniciales de descarga NLTK o reinstala `nltk`.
- **CSV ParserError / unexpected end of data**: usa `engine="python"` + `on_bad_lines="skip"` o revisa el archivo (a veces está truncado).
- **AccessDeniedException (Comprehend/Translate)**: el rol no tiene permisos → usa `NLP_PROVIDER=local` o solicita policy IAM.
- **HuggingFace tarda mucho la primera vez**: está descargando el modelo; luego queda cacheado en `.hf_cache/`.

## Estructura del Repositorio
```
├─ nlp_pipeline_aws_local.ipynb
├─ README.md
├─ requirements.txt          
└─ data/
   └─ AMAZON-REVIEW-DATA-CLASSIFICATION.zip 
```

## Créditos y referencias
```md
Este proyecto fue inspirado/estructurado a partir de contenidos de:

- AWS Academy — Machine Learning for Natural Language Processing (curso base).
  - https://aws.amazon.com/training/awsacademy/
- AWS Academy Graduate — Machine Learning for Natural Language Processing (Credly): https://www.credly.com/badges/d008f061-908e-48bb-ae79-257d03fe53ce

Servicios y documentación utilizados:

- Amazon Comprehend (NLP managed service en AWS).
  - https://aws.amazon.com/comprehend/
- Amazon Translate (traducción neural EN→ES opcional).
  - https://aws.amazon.com/translate/
  - API: TranslateText — https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html

Baselines y librerías en modo local:

- NLTK VADER (sentiment baseline).
  - https://www.nltk.org/api/nltk.sentiment.vader.html
- scikit-learn TF-IDF (features + key phrases aproximadas).
  - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- HuggingFace / MarianMT (traducción local fallback con Helsinki-NLP/opus-mt-en-es).
  - https://huggingface.co/Helsinki-NLP/opus-mt-en-es
```
