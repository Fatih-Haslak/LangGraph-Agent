# Industrial Multi-Agent System with LangGraph

> Endüstriyel seviyede Türkçe çok-ajanlı yapay zeka sistemi — LangGraph + Turkish-Gemma-9b

**v2.0** · Python 3.10+ · LangGraph · HuggingFace Transformers

---

## Özellikler

- 🗺️ **Planner Agent** — Sorguyu analiz eder, görev tipi ve adımları belirler (JSON plan)
- 🔀 **Router Agent** — Plana göre iş akışını doğru ajana yönlendirir
- 📚 **Wikipedia Entegrasyonu** — Türkçe Wikipedia'da iki aşamalı arama ve içerik çekme
- 📝 **Summarizer Agent** — Uzun Wikipedia içeriğini soruya odaklı yoğunlaştırır
- 🧮 **Math Agent** — Adım adım matematiksel hesaplama
- 💬 **Chat Agent** — Konuşma geçmişine duyarlı doğal Türkçe sohbet
- 🔍 **QA Checker Agent** — 0-10 kalite skoru + otomatik yeniden deneme döngüsü
- 🧠 **Conversation Memory** — Kayan pencereli konuşma geçmişi (son 6 tur)
- ⚙️ **System Logger** — Zaman damgalı yapılandırılmış yürütme izi
- ⚡ **4-bit Kuantizasyon** — BitsAndBytes NF4 ile verimli VRAM kullanımı

---

## Mimari

```
INPUT
  ↓
🗺️  PlannerAgent       → JSON plan üretir (task_type, steps, intent…)
  ↓
🔀  RouterAgent        → research | math | chat
  ↓
  ┌── 📚 WikiSearchAgent  → wiki_raw{}
  │      ↓
  │   📝 SummarizerAgent  → wiki_summary  [koşullu]
  │
  ├── 🧮 MathAgent        → math_result
  │
  └── 💬 ChatAgent        → (pass-through)
       ↓
✍️  AnswerGeneratorAgent  → draft_answer
  ↓
🔍  QualityCheckerAgent   → qa_report + final_answer
  │
  ├── skor ≥ 6 → END ✅
  └── skor < 6 → answer_gen (max 2 retry) 🔄
```

Tam görsel dokümantasyon: [`docs/system_architecture.html`](docs/system_architecture.html)

---

## Kurulum

### Gereksinimler

- Python 3.10+
- CUDA destekli GPU (önerilen)
- 8 GB+ VRAM (4-bit kuantizasyon ile)

### Bağımlılıklar

```bash
git clone https://github.com/yourusername/LangGraph-Agent.git
cd LangGraph-Agent
pip install -r requirements.txt
```

---

## Kullanım

```bash
python agent.py
```

### Örnek Etkileşimler

```
🧑  Sorgu: Atatürk kimdir?
  🗺️ [PLANNER   ] Plan → task_type: research, requires_summary: true
  🔀 [ROUTER    ] research → wiki
  📚 [WIKI      ] Bulundu: 'Mustafa Kemal Atatürk' (4200 karakter)
  📝 [SUMMARIZER] Özet üretildi: 480 karakter
  ✍️  [ANSWER    ] Araştırma cevabı üretildi
  🔍 [QA        ] Kalite skoru: 8/10 | Onaylı: True
🤖  Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusu...

🧑  Sorgu: 144'ün karekökü kaçtır?
  🗺️ [PLANNER   ] Plan → task_type: math
  🔀 [ROUTER    ] math → math
  🧮 [MATH      ] Hesaplama tamamlandı
  🔍 [QA        ] Kalite skoru: 9/10 | Onaylı: True
🤖  √144 = 12

🧑  Sorgu: Merhaba!
  🗺️ [PLANNER   ] Plan → task_type: chat
  💬 [CHAT      ] → answer_gen
  🔍 [QA        ] Kalite skoru: 7/10 | Onaylı: True
🤖  Merhaba! Size nasıl yardımcı olabilirim?
```

---

## Teknik Detaylar

### Model

| Parametre | Değer |
|---|---|
| Model | `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` |
| Kuantizasyon | 4-bit BitsAndBytes (NF4, double_quant) |
| compute_dtype | `torch.float16` |
| Framework | PyTorch + HuggingFace Transformers |

### AgentState Alanları

| Alan | Tip | Açıklama |
|---|---|---|
| `session_id` | str | UUID v4 tabanlı 8 karakter oturum kimliği |
| `user_query` | str | Ham kullanıcı sorgusu |
| `conversation_history` | List | ConversationMemory'den gelen geçmiş |
| `plan` | Dict | PlannerAgent çıktısı |
| `wiki_raw` | Dict | Ham Wikipedia verisi |
| `wiki_summary` | str | Özetlenmiş Wikipedia içeriği |
| `math_result` | str | Matematiksel hesaplama çıktısı |
| `draft_answer` | str | QA öncesi taslak cevap |
| `final_answer` | str | QA onaylı nihai cevap |
| `qa_report` | Dict | Kalite raporu (skor, sorunlar, gerekçe) |
| `retry_count` | int | Yeniden deneme sayacı (max: 2) |
| `execution_trace` | List | Adım adım yürütme log kayıtları |
| `error_log` | List | Araç hataları |

### Sistem Limitleri

| Sabit | Değer |
|---|---|
| `MAX_RETRY_ATTEMPTS` | 2 |
| `QUALITY_THRESHOLD` | 6 / 10 |
| `MEMORY_WINDOW` | 6 tur |
| `WIKI_SUMMARY_MAX_CHARS` | 1200 |
| `WIKI_CONDENSED_MAX_CHARS` | 500 |

---

## Performans

RTX 4060 Ti SUPER, 4-bit kuantizasyon:

| Ajan | Ortalama Süre | Not |
|---|---|---|
| Chat Agent | ~7-8 s | Doğrudan LLM çıkarımı |
| Math Agent | ~9-11 s | LLM tabanlı hesaplama |
| Wiki Agent (özetli) | ~35-45 s | Wikipedia API (~15-20 s) + Summarizer |
| Wiki Agent (özetsiz) | ~30-35 s | Wikipedia API + Answer Gen |

**VRAM Kullanımı:** ~8 GB (4-bit kuantizasyon)

---

## Proje Yapısı

```
LangGraph-Agent/
├── agent.py                    # Ana sistem (tüm ajanlar)
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Bu dosya
└── docs/
    └── system_architecture.html  # Görsel mimari dokümantasyon
```

---

## Kaynaklar

- [Turkish-Gemma-9b — YTU CE Cosmos Lab](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1)
- [LangGraph — LangChain](https://github.com/langchain-ai/langgraph)
- [Türkçe Wikipedia API](https://tr.wikipedia.org/w/api.php)
- [Cosmos Lab](https://cosmos.yildiz.edu.tr/)
