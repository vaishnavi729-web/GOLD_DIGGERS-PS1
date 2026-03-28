# Green-Truth Auditor 🍃

An AI-powered greenwashing detection tool for hackathons. Focuses on speed, reliability, and technical depth with 3 local AI models and a stunning Glassmorphism UI.

## 🚀 Key Features

1.  **Multi-Modal Input**: Analyze product via text, URL scraping, or image OCR.
2.  **ClimateBERT Integration**: Classifies claims as "Marketing Fluff" vs "Evidence-Based".
3.  **Vague-to-Value Score**: A weighted 0-100 score based on 4 distinct factors.
4.  **Semantic Contradiction**: Detects if "natural" claims clash with "synthetic" ingredients.
5.  **Certified Alternatives**: Recommends similar products from B-Corp/GOTS verified brands.
6.  **Crowdsourced Reports**: Community-driven reporting of false claims.
7.  **Smart Chatbot**: Context-aware assistant to explain scores and buzzwords.
8.  **Premium UI**: Glassmorphism, particles, score gauge animations, and dark mode.

## 🛠️ Tech Stack

-   **Backend**: Flask, BeautifulSoup4, SQLite, Pandas.
-   **AI Models**:
    -   `climatebert/distilroberta-base-climate-fever` (Claim classification)
    -   `all-MiniLM-L6-v2` (Semantic similarity & contradiction)
    -   `EasyOCR` (Fast image-to-text extraction)
-   **Frontend**: Tailwind CSS, GSAP, Vanilla JS, Font Awesome.

## 📋 Prerequisites

-   Python 3.10+
-   At least 8GB RAM (for loading ML models)
-   Windows/macOS/Linux

## 📥 Installation

1.  **Clone or create the directory**:
    ```bash
    mkdir green-truth-auditor
    cd green-truth-auditor
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will download several hundred MBs of model weights on the first run.*

3.  **Run the application**:
    ```bash
    python app.py
    ```

4.  **Open in Browser**:
    Navigate to `http://127.0.0.1:5000`

## 🧪 Testing Examples

-   **Text**: "Our 100% natural, eco-friendly plastic bottle is biodegradable and organic."
-   **URL**: Paste a link to a product page with sustainability claims (e.g., Patagonia or Allbirds).
-   **Image**: Upload a clear photo of a laundry detergent or apparel label.

## ⚠️ Troubleshooting

-   **Model Load Error**: If the app fails to start, ensure you have a stable internet connection for the first run (to download models from HuggingFace).
-   **OCR Slow**: EasyOCR may take 5-10 seconds on CPUs without a GPU.
-   **Memory Usage**: If the app crashes, close other heavy applications to free up RAM.

## 📜 Credits
Built for the Hackathon 
  prototype video
  https://drive.google.com/file/d/1HRcTzujedpzC-QG3hDPW72uZeVpAunwF/view?usp=drivesdk
